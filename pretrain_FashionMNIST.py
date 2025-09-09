import torch
import torchhd as hd
import torch.nn as nn
from args import args
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import HD_encoder, binarize_hdrp, LeHDC
from torchhd.embeddings import Random, Level, Projection, Sinusoid, Density


def train(model, train_loader, optimizer, device, hd_rand_proj, args):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch in train_loader:
        optimizer.zero_grad()
        raw_feature, label = batch

        label = label.long().to(device[0])
        raw_feature = raw_feature.to(device[0])

        HV_feature = HD_encoder(raw_feature, hd_rand_proj)

        outputs, _, _ = model(HV_feature)
        loss = nn.CrossEntropyLoss()(outputs, label)

        loss.backward()
        if args.maxL2 is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.maxL2)  # 梯度裁剪, nax_norm规定最大的L2范数
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == label).sum().item()
        total_preds += label.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_preds / total_preds

    return epoch_loss, epoch_acc


def validate(model, val_loader, device, hd_rand_proj, args):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    correct_preds_hd = 0
    total_preds = 0

    with torch.no_grad():
        for batch in val_loader:
            raw_feature, label = batch

            label = label.long().to(device[0])
            raw_feature = raw_feature.to(device[0])

            HV_feature = HD_encoder(raw_feature, hd_rand_proj)

            outputs, query_vectors, associate_memory = model(HV_feature)
            loss = nn.CrossEntropyLoss()(outputs, label)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == label).sum().item()

            similarity_metrics = hd.hamming_similarity(query_vectors, associate_memory)
            _, predicted_hd = torch.max(similarity_metrics, 1)
            correct_preds_hd += (predicted_hd == label).sum().item()

            total_preds += label.size(0)

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct_preds / total_preds
        epoch_acc_hd = correct_preds_hd / total_preds

    return epoch_loss, epoch_acc, epoch_acc_hd


def main():
    if args.multigpu is None:
        device = torch.device("cpu")
    elif len(args.multigpu) == 1:
        device = [torch.device(f'cuda:{args.multigpu[0]}')]
    else:
        device = [torch.device(f'cuda:{gpu_id}') for gpu_id in args.multigpu]

    print(f"Using device(s): {device}")

    # ------------------- get dataset -------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_val_ds = datasets.FashionMNIST(root='.', train=True,
                                         download=True, transform=transform)

    val_size = int(0.05 * len(train_val_ds))  # 5 % 验证
    train_size = len(train_val_ds) - val_size  # 95 % 训练

    train_ds, val_ds = random_split(train_val_ds, [train_size, val_size])

    train_loader = DataLoader(train_val_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(datasets.FashionMNIST(root='.', train=False, download=True, transform=transform),
                             batch_size=args.batch_size)
    sample_img, sample_label = train_ds[0]
    n_features = sample_img.numel()
    n_classes = len(train_val_ds.classes)

    # ------------------- HDC encoder -------------------
    hd_rand_proj = Projection(n_features, args.HV_dim, requires_grad=False, device=device[0])
    hd_rand_proj = binarize_hdrp(hd_rand_proj, device=device[0])
    torch.save(hd_rand_proj, 'hd_rand_proj.pth')

    if len(args.multigpu) > 1:
        model = nn.DataParallel(LeHDC(n_dimensions=args.HV_dim, n_classes=n_classes, dropout=args.dropout),
                                device_ids=device)
    else:
        model = LeHDC(n_dimensions=args.HV_dim, n_classes=n_classes, dropout=args.dropout)
    model = model.to(device[0])

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, cooldown=0,
                                  verbose=True, min_lr=1e-7)

    num_epochs = args.epoch
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, device, hd_rand_proj, args)
        # val_loss, val_acc, _ = validate(model, val_loader, device, hd_rand_proj, args)

        scheduler.step(train_loss)

        # # save model
        # if isinstance(model, nn.DataParallel):
        #     torch.save(model.module.state_dict(), f'pretrain_model_epoch_{epoch + 1}.pth')
        # else:
        #     torch.save(model.state_dict(), f'pretrain_model_epoch_{epoch + 1}.pth')

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        # print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    print(args)
    main()
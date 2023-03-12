import os
import tsp


def main():
    knn = [5, 10]
    dataset = tsp.dataset.TSPDataset(
        transforms=tsp.dataset.transforms.Compose([
            tsp.dataset.transforms.MaxNodesTransform(100),
            tsp.dataset.transforms.KNNTransform(knn),
        ]),
    )
    device = "cuda" # or "cpu"
    path_to_training_config = "configs/training.yml"
    model_folder_name = "conv_net"

    # Model initialization
    model = tsp.utils.make_model(
        path_to_training_config, knn_dim=len(knn), model_name=model_folder_name)
    
    # Trainer initialization
    trainer, chkp_clbk = tsp.utils.make_trainer(path_to_training_config, name=model_folder_name, return_chkp_clbk=True)

    # Training and validation loaders initialization
    train_loader, val_loader = tsp.utils.make_train_val_loaders(path_to_training_config, dataset)

    # Training
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    WORKDIR = "/Users/shpakovych/repos/tsp/examples/training"

    is_set_workdir = True
    if is_set_workdir:
        os.chdir(WORKDIR)

    main()
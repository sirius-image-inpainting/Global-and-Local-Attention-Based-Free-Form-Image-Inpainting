from DataModule.PlacesDataModule import PlacesDataModule

from argparse import ArgumentParser, Namespace

from pytorch_lightning.trainer import Trainer

def main(args: Namespace) -> None:
    model =
    datamodule = PlacesDataModule()

    trainer = Trainer.from_argparse_args()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser =
    places_datamodule = PlacesDataModule()

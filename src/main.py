import click

from deepSVDD import DeepSVDD
from datasets.main import load_dataset


@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10']))
@click.argument('model_name')
def main(dataset_name, model_name):

    # Load data
    dataset = load_dataset(dataset_name)

    # Initialize DeepSVDD with model
    # deep_SVDD = DeepSVDD(objective='soft-boundary', nu=0.01)
    deep_SVDD = DeepSVDD()
    deep_SVDD.set_model(model_name)

    # Pretrain model on datasets (via autoencoder)
    deep_SVDD.pretrain(dataset, optimizer_name='adam', lr=0.001, n_epochs=50, batch_size=200)

    # Train model on dataset
    deep_SVDD.train(dataset, optimizer_name='adam', lr=0.0001, n_epochs=25, batch_size=200)
    deep_SVDD.test(dataset)


if __name__ == '__main__':
    main()

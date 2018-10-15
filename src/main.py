import click

from deepSVDD import DeepSVDD


@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10']))
@click.argument('model_name')
def main(dataset_name, model_name):

    deep_SVDD = DeepSVDD(dataset_name, model_name)


if __name__ == '__main__':
    main()

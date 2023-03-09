# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    from torch.utils.data import DataLoader
    from core.utils import ImageDataset
    import matplotlib.pyplot as plt

    image_data = ImageDataset()
    loader = DataLoader(image_data, batch_size=1, shuffle=False)
    for batch in loader:
        image = batch[0]
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
        break


    #
    # plt.imshow(image.permute(1, 2, 0))
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np

def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, dtype=np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dir_path = "offline_dpo_trl/dpo_loss_log.txt"
    y_train_loss = data_read(dir_path)

    x_train_loss = range(len(y_train_loss))

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel("Training Steps")
    plt.ylabel("DPO Loss")

    plt.plot(x_train_loss, y_train_loss, linewidth=1,linestyle = "solid",color='blue', label='DPO Loss')

    plt.legend()
    plt.title("DPO Training Loss Curve")
    plt.savefig("./offline_dpo_trl/dpo_loss_curve.png", dpi=300)
from urllib.request import urlopen


@profile
def main():
    response = urlopen("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz")
    CHUNK = 16 * 1024
    with open("k49-train-imgs.npz", "wb") as f:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)


if __name__ == "__main__":
    main()


def view_training(model, loader, display):
    images, labels = next(iter(loader))
    num_images = min(images.size(0), 4)

    outputs = model(images.to(device))
    outputs = outputs.cpu().detach()
    images, labels = images.cpu(), labels.cpu()
    output_labels = torch.argmax(outputs.cpu().detach(), dim=1)

    if display is False:
        plt.ioff()
    fig, axes = plt.subplots(4, num_images, figsize=(3 * num_images, 8))
    time.sleep(1)
    for i in range(num_images):
        if num_images > 1:
            ax0 = axes[0, i]
            ax1 = axes[1, i]
            ax2 = axes[2, i]
            ax3 = axes[3, i]
        else:
            ax0 = axes[0]
            ax1 = axes[1]
            ax2 = axes[2]
            ax3 = axes[3]
        ax0.axis('off')
        ax0.set_title('Image')
        ax0.imshow(images[i].permute(1, 2, 0))
        ax1.axis('off')
        ax1.set_title('Label')
        ax1.imshow(labels[i].permute(1, 2, 0))
        ax2.axis('off')
        ax2.set_title('Output (prob)')
        ax2.imshow(outputs[i].permute(1, 2, 0))
        ax3.axis('off')
        ax3.set_title('Output (argmax)')
        ax3.imshow(output_labels[i])
    plt.tight_layout()
    plt.show()
    date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
    plt.savefig(params['script_dir'] + '/Output/labels' + date_str + '.png')
    plt.close()


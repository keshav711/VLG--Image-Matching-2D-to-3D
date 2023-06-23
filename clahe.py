def to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])


def convert_clahe(initial):
    """
    Convert an image using Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters:
        initial (ndarray): The input image as a NumPy array.

    Returns:
        ndarray: The CLAHE-enhanced version of the input image.

    Description:
        This function applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the local contrast
        of an input image. The CLAHE algorithm redistributes pixel intensities in a way that improves the visibility
        of local details while preventing excessive contrast enhancement.

        The input image should be in BGR color format. The function performs the following steps:

        1. Create a CLAHE object with a clipLimit of 2.0 and a tileGridSize of (8,8).
        2. Convert the input image to the HSV color space.
        3. Extract the intensity (value) channel from the HSV image.
        4. Apply the CLAHE algorithm to the intensity channel.
        5. Convert the image back to the BGR color space.
        6. Return the CLAHE-enhanced image.

    Example:
        # Load an image
        image = cv2.imread('input.jpg')

        # Apply CLAHE conversion
        clahe_image = convert_clahe(image)

        # Display the result
        cv2.imshow('CLAHE-enhanced Image', clahe_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ac = np.asarray(initial)
    ac = cv2.cvtColor(ac, cv2.COLOR_BGR2HSV)
    ac[:,:,-1] = clahe.apply(ac[:,:,-1])
    ac = cv2.cvtColor(ac, cv2.COLOR_HSV2BGR)
    return ac


def imageof(file_name, device=torch.device('cpu')):
    """
    Load an image file using OpenCV and convert it to a torch tensor.

    Parameters:
        file_name (str): The path to the image file.
        device (torch.device, optional): The device to which the torch tensor will be moved. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The loaded image as a torch tensor.

    Description:
        This function loads an image file using OpenCV and converts it to a torch tensor, which is suitable for further
        processing in a PyTorch-based model.

        If the `APPLY_CLAHE` flag is set to `True`, the function applies Contrast Limited Adaptive Histogram Equalization
        (CLAHE) to enhance the local contrast of the image before conversion. Otherwise, the image is loaded without any
        additional preprocessing.

        The image is loaded as a BGR image using OpenCV's `cv2.imread()` function. It is then converted to a torch tensor
        and normalized by dividing by 255. The image is further converted from the BGR color space to the RGB color space
        using Kornia's `color.bgr_to_rgb()` function.

        Finally, the resulting torch tensor is moved to the specified device before being returned.

    Example:
        # Load and preprocess an image
        file_path = 'image.jpg'
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        image = load_torch(file_path, device)

        # Perform further processing using the torch tensor
        output = model(image)
    """
    if APPLY_CLAHE:
        image_ = convert_clahe(cv2.imread(file_name))
    else:
        image_ = cv2.imread(file_name)

    image_ = K.image_to_tensor(image_, False).float() / 255.
    image_ = K.color.bgr_to_rgb(image_.to(device))
    return image_

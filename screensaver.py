import cv2
import multiprocessing
import numpy as np
import PIL.ImageFile
import PIL.Image

from pathlib import Path
from scipy import interpolate
from typing import Sequence


def get_positions_and_angles(
    image_size: Sequence[int], screen_res: Sequence[int], tlp: Sequence[Sequence[int]], brp: Sequence[Sequence[int]],
) -> tuple[np.ndarray]:
    """
    When choosing which subsections within the original image will appear in the screensaver, the user uses the
    coordinates of the top-left and bottom-right corners of these rectangular subsections as reference points.  To make
    things simpler, only the top-left corner need be precise; this is accomplished through the function
    "get_positions_and_angles", which uses the bottom-right coordinates to determine the orientation of the rectangle.
    Given that the dimensions of the final screensaver are given by the user, the top-left corner and the extrapolated
    angle are all that is needed to complete this process.

        === Inputs ===

    image_size      Sequence of two integers representing the resolution of the original image in pixels; (w, h).
    screen_res      Sequence of two integers representing monitor size in pixels; e.g. (1920, 1080) or (3440, 1440).
    tlp             2-D sequence containing integers.  Shape should be (N, 2). Top-left points.
    brp             2-D sequence containing integers.  Shape should be (N, 2). Bottom-right points.

        === Outputs ===

    tlp             numpy.ndarray<int64>; a 2-D array of integers [shape (N, 2)]. Top-left points.
    angles          numpy.ndarray<float64>; a 1-D set of image rotation angles of type float measured in degrees.
    """
    # Converting tlp (the top-left coordinates of each selected rectangular subsection) to a NumPy array.
    tlp = np.array(tlp, dtype=np.int64)
    # Converting brp (the bottom-right coordinates of each selected rectangular subsection) to a NumPy array.
    brp = np.array(brp, dtype=np.int64)
    # The angles (in radians) between the top-left and bottom-right of each selected rectangle.
    default_angle = np.arctan2(screen_res[1], screen_res[0])
    # The angles at which the image should be rotated prior to cropping.
    angles = np.arctan2(brp[:, 1] - tlp[:, 1], brp[:, 0] - tlp[:, 0]) - default_angle
    # Returning the top-left coordinates and rotation angles.
    return tlp, np.rad2deg(angles)


def interpolate_tlp(tlp: Sequence[Sequence[int]], angles: Sequence[float], total_frames: int) -> tuple[np.ndarray]:
    """
    Since the user only selects discrete rectangular subsections within the original image, a curved path connecting
    these points is needed in order to create a smooth animation that will pan and rotate across the original image.
    Given the output of "get_positions_and_angles", this function "interpolate_tlp" uses class
    InterpolatedUnivariateSpline from SciPy's interpolate sub-package to create points in between those given in arrays
    "tlp" and "angles".

        === Inputs ===

    tlp             2-D sequence containing integers.  Shape should be (N, 2). Top-left points.
    angles          1-D sequence of image rotation angles of type float measured in degrees.
    total_frames    Integer. The number of frames that will appear in the final screensaver video file.

        === Outputs ===

    tlp             numpy.ndarray<int64>; a 2-D array of integers [shape (total_frames, 2)]. Top-left points.
    angles          numpy.ndarray<float64>; a 1-D set of image rotation angles of type float measured in degrees.
    """
    # Converting tlp (the top-left coordinates of each selected rectangular subsection) to a NumPy array.
    tlp = np.array(tlp, dtype=np.int64)
    # Creating a 1-D array for the parametric variable "t" as an input in "InterpolatedUnivariateSpline".
    t = np.linspace(0, 1, tlp.shape[0], dtype=np.float64)
    # Creating a function that uses the parametric value "t" to extrapolate an x-coordinate.
    x_func = interpolate.InterpolatedUnivariateSpline(t, tlp[:, 0], k=2)
    # Creating a function that uses the parametric value "t" to extrapolate a y-coordinate.
    y_func = interpolate.InterpolatedUnivariateSpline(t, tlp[:, 1], k=2)
    # Creating a function that uses the parametric value "t" to extrapolate an angle.
    theta_func = interpolate.InterpolatedUnivariateSpline(t, angles, k=min(tlp.shape[0] - 1, 5))
    # Creating a 1-D of parametric values "t" to generate the data between the user-defined rectangles.
    t = np.linspace(0, 1, int(total_frames))
    # Preparing the full array of top-left points.
    tlp = np.array([x_func(t), y_func(t)]).T
    # Returning the full array of top-left coordinates and the full array of image angles.
    return tlp, theta_func(t)


def get_rotated_coordinates(
    image_size: Sequence[int], screen_res: Sequence[int], tlp: Sequence[Sequence[int]], angles: Sequence[float],
) -> np.ndarray:
    """
    Once the full set of rectangle coordinates has been extrapolated in function "interpolate_tlp", a new problem arises
    in that the first step in obtaining the set of cropped images is to rotate the original image to a new orientation,
    possibly once every frame.  This means that a coordinate transformation is needed in order to convert the points
    in "tlp" to their new rotated coordinate systems.  This is accomplished by generating a set of rotation matrices
    (one for each frame/angle) and performing matrix multiplication between these matrices and the vectors pointing to
    each upper-left coordinate.

        === Inputs ===

    image_size      1-D sequence containing two integers, representing the size in pixels of the original image.
    screen_res      1-D Sequence of two integers representing monitor size in pixels; e.g. (1920, 1080) or (3440, 1440).
    tlp             2-D sequence containing integers.  Shape should be (N, 2). Top-left points.
    angles          1-D sequence of image rotation angles of type float measured in degrees. Shape should be (N,)

        === Outputs ===

    coordinates     numpy.ndarray<int64>; a 2-D array of integers [shape (total_frames, 4)]. Top-left and bottom-right
                    rectangle coordinates that take the image orientation into account.
    """
    # Converting the angles from radians into degrees.
    angles = np.deg2rad(angles)
    # Creating the clockwise rotation matrices. To make these usable in matrix multiplication, they must be transposed.
    rotation_matrices = np.array([[np.cos(angles), np.sin(angles)], [-np.sin(angles), np.cos(angles)]]).T
    # After being transposed, the matrices must also be rotated twice around axis zero.
    rotation_matrices = np.rot90(rotation_matrices, k=2, axes=(1, 2))
    # Getting the center coordinates of the original image.
    center = (image_size[0] / 2, image_size[1] / 2)
    # Subtracting the center from the set of coordinates -- this is required for rotations about the image's center.
    tlp = np.array(tlp)[:, :, None] - np.array(center)[:, None]
    # Performing the matrix multiplication, which executes the coordinate transformation.
    transformed = (rotation_matrices @ tlp)[:, :, 0]
    # Calculating the center coordinates of each new image, since the rotations change its dimensions.
    new_center = np.array(
        [
            np.abs(image_size[0] * np.cos(angles) / 2) + np.abs(image_size[1] * np.sin(angles) / 2),
            np.abs(image_size[0] * np.sin(angles) / 2) + np.abs(image_size[1] * np.cos(angles) / 2),
        ]
    ).T
    # Re-adding the new center coordinates to the rectangle's points, since the origin is at matrix coordinate (0, 0).
    transformed = transformed + new_center
    # Preparing the array which will be used to actually extract the sub-rectangles and create the final video.
    coordinates = np.array(
        [transformed[:, 0], transformed[:, 1], transformed[:, 0] + screen_res[0], transformed[:, 1] + screen_res[1],]
    ).T
    # Returning the final set of coordinates.
    return coordinates


def initializer(image: PIL.ImageFile.ImageFile) -> None:
    """
    Since the original image may be large, we wish to prevent it being copied multiple times when using the built-in
    multiprocessing python package.  We can create an initializer function that allows the original image to be shared
    among all processor cores, and pass it into an instance of multiprocessing.Pool to accomplish this.

        === Inputs ===

    image           An image file (e.g. "PIL.PngImagePlugin.PngImageFile") from the PIL package.

        === Outputs ===

    None

    """
    # Declaring that variable "img" will be global
    global img
    # Setting the value of "img" to the one given in parameter "image".
    img = image


def worker(data: tuple[Sequence[int], float]) -> PIL.ImageFile.ImageFile:
    """
    Worker function passed to the imap method in an instance of multiprocessing.Pool that implements image rotation and
    cropping using the coordinates and angles calculated in the functions "get_positions_and_angles", "interpolate_tlp",
    and "get_rotated_coordinates".

        === Inputs ===

    data            A tuple containing a 1-D sequence of four integers (the upper-left and bottom-right coordinates of a
                    rectangle) and a single float representing the image's rotation angle.

        === Outputs ===

    image           An image file (e.g. "PIL.PngImagePlugin.PngImageFile") from the PIL package.
    """
    coordinate, angle = data
    global img
    rotated = img.rotate(angle=angle, expand=True)
    image = rotated.crop(coordinate)
    return image


def run_parallel(
    video_name: str,
    fps: int,
    screen_res: Sequence[int],
    image: PIL.ImageFile.ImageFile,
    coordinates: Sequence[Sequence[int]],
    angles: Sequence[float],
    total_frames: int,
) -> None:
    """
    A function that manages all the parallelization, transformation, and video writing operations.  Additionally, will
    print out the percentage progress of the video writing process to the terminal.

        === Inputs ===

    video_name      The name (string) of the video file where the screensaver will be written.
    fps             The (integer-valued) frames-per-second of the video.
    screen_res      1-D Sequence of two integers representing monitor size in pixels; e.g. (1920, 1080) or (3440, 1440).
    image           An image file (e.g. "PIL.PngImagePlugin.PngImageFile") from the PIL package.
    coordinates     numpy.ndarray<int64>; a 2-D array of integers [shape (total_frames, 4)]. Top-left and bottom-right
                    rectangle coordinates that take the image orientation into account.
    angles          1-D sequence of image rotation angles of type float measured in degrees. Shape should be (N,)
    total_frames    Integer. The number of frames that will appear in the final screensaver video file.

        === Outputs ===

    None
    """
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video = cv2.VideoWriter(video_name, fourcc, fps, screen_res)

    perc = 0
    processes = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(processes=processes, initializer=initializer, initargs=(image,))
    perc_decimals = 1

    for n, image in enumerate(pool.imap(worker, ((i, j) for i, j in zip(coordinates, angles)))):
        new_perc = round(100 * n / total_frames, 1)
        if new_perc > perc:
            print(f"\r    Loading: {perc:.{perc_decimals}f}%", end="", flush=True)
            perc = new_perc
        video.write(cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    video.release()
    pool.close()
    print(f"\r    Loading: {100:.{perc_decimals}f}%")


def run(
    image_name: str,
    video_name: str,
    screen_res: Sequence[int],
    tlp: Sequence[Sequence[int]],
    brp: Sequence[Sequence[int]],
    total_frames: int,
    fps: int = 60,
    skip_last: bool = False,
) -> None:
    """
    This is the main program function -- users who are not interested in the program's inner workings should be able to
    control most program features from here.  In order to use this program, the user must collect two sequences of
    points located within the original image (measured in pixels).  The first sequence ("tlp") represents the upper-left
    corners of the rectangles which will be used to select subsections of the original image, while the second ("brp")
    represents the rectangles' bottom right coordinates.  The sequence "brp" does not need to be accurate, as the
    program automatically determines the correct rectangle proportions based on the given "screen_res".

        === Inputs ===

    image_size      Sequence of two integers representing the resolution of the original image in pixels; (w, h).
    video_name      The name (string) of the video file where the screensaver will be written.
    screen_res      Sequence of two integers representing monitor size in pixels; e.g. (1920, 1080) or (3440, 1440).
    tlp             2-D sequence containing integers.  Shape should be (N, 2). Top-left points.
    brp             2-D sequence containing integers.  Shape should be (N, 2). Bottom-right points.
    total_frames    Integer. The number of frames that will appear in the final screensaver video file.
    fps             The (integer-valued) frames-per-second of the video.
    skip_last       If the video is meant to be a perfect loop, set to True in order to skip the last frame.

        === Outputs ===

    None
    """
    PIL.Image.MAX_IMAGE_PIXELS = None
    print("[1] Opening Image")
    image = PIL.Image.open(image_name)
    screen_res = np.array(screen_res)
    image_size = np.array(image.size)

    print("[2] Correcting Rectangle Positions")
    pos, angles = get_positions_and_angles(image_size, screen_res, tlp, brp)
    print("[3] Interpolating Rectangle Positions and Orientations")
    pos, angles = interpolate_tlp(pos, angles, total_frames)
    print("[4] Calculating Coordinate Shift and Rotation")
    coordinates = get_rotated_coordinates(image_size, screen_res, pos, angles)

    if skip_last:
        coordinates = coordinates[:-1]

    if not video_name.lower().endswith(".mp4"):
        video_name += ".mp4"

    print(f"[5] Writing Series of Images to Video File: {video_name}")
    images = run_parallel(video_name, fps, screen_res, image, coordinates, angles, total_frames)


if __name__ == "__main__":

    """
    The image used in this example is the full-resolution version of Cosmic Cliffs, one of the first photos taken by the
    James Webb Space Telescope.

    Dimensions:     14575 x 8441 pixels
    Info URL:       https://webbtelescope.org/contents/media/images/2022/031/01G77PKB8NKR7S8Z6HBXMYATGJ?news=true
    Direct URL:     https://stsci-opo.org/STScI-01G7WCHVJ1VXPW5CX5DSVE0W1F.png
    """

    # The name of the file from the direct URL above. Should be placed in the directory containing this program.
    image_name = "STScI-01G7WCHVJ1VXPW5CX5DSVE0W1F.png"

    # The currently selected screen resolution is for ultrawide 1440p monitors.
    screen_res = (3440, 1440)

    # Selects a video name based on the chosen resolution.
    video_name = f"cosmic_cliffs_screensaver_{screen_res[0]}_{screen_res[1]}"

    # The total number of frames in the final video.
    total_frames = 12500

    # The framerate of the final video.
    fps = 60

    # Top Left Positions -- hand picked for the selected resolution and image dimensions.
    top_left_positions = [
        [1000, 1003.81],
        [3500, 2750],
        [6700, 4450],
        [10600, 2600],
        [8600, 3500],
        [6200, 3300],
        [3500, 3500],
        [950, 965.71],
        [1000, 1003.81],
    ]

    # Bottom Right Positions (Precision Unimportant) -- hand picked for the selected resolution and image dimensions.
    bottom_right_positions = [
        [3072, 4104.65],
        [7580, 4480],
        [9700, 2300],
        [14300, 4500],
        [6500, 6600],
        [4700, 6300],
        [3500, 5500],
        [3082, 4288],
        [3272, 4304.65],
    ]

    # Create the screensaver
    run(
        image_name,
        video_name,
        screen_res,
        top_left_positions,
        bottom_right_positions,
        total_frames,
        fps,
        skip_last=True,
    )

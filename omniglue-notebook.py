{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# OmniGlue - Feature Matching with Foundation Model Guidance\n",
    "\n",
    "This notebook will help you set up and test the OmniGlue library from Google Research, which is designed for generalizable image feature matching using foundation model guidance.\n",
    "\n",
    "OmniGlue was introduced in a CVPR 2024 paper as a solution for image matching that can better generalize to novel image domains not seen during training."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Environment Setup\n",
    "\n",
    "First, let's set up the environment by installing the necessary packages. We'll create a conda environment, clone the repository, and install the required dependencies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Install OmniGlue and its dependencies\n",
    "!git clone https://github.com/google-research/omniglue.git\n",
    "%cd omniglue\n",
    "!pip install -e ."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Download Required Models\n",
    "\n",
    "OmniGlue requires multiple pre-trained models to work properly:\n",
    "1. SuperPoint - For keypoint detection\n",
    "2. DINOv2 - A vision foundation model (vit-b14)\n",
    "3. OmniGlue weights - The trained OmniGlue model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create models directory\n",
    "!mkdir -p models\n",
    "%cd models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Download SuperPoint\n",
    "!git clone https://github.com/rpautrat/SuperPoint.git\n",
    "!mv SuperPoint/pretrained_models/sp_v6.tgz .\n",
    "!rm -rf SuperPoint\n",
    "!tar zxvf sp_v6.tgz\n",
    "!rm sp_v6.tgz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Download DINOv2 (vit-b14)\n",
    "!wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Download OmniGlue weights\n",
    "!wget https://storage.googleapis.com/omniglue/og_export.zip\n",
    "!unzip og_export.zip\n",
    "!rm og_export.zip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Go back to the main directory\n",
    "%cd .."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Import Libraries\n",
    "\n",
    "Now let's import the required libraries for testing OmniGlue."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import omniglue\n",
    "from omniglue import utils\n",
    "from PIL import Image\n",
    "import cv2\n",
    "import time"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Download Test Images\n",
    "\n",
    "Let's download some sample images to test OmniGlue or use the demo images from the repository."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Check if demo images exist in the repo\n",
    "!ls -la res/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create a directory for our own test images if needed\n",
    "!mkdir -p test_images"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Define OmniGlue Matching Function\n",
    "\n",
    "Let's create a function that performs image matching using OmniGlue based on the demo script."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "def match_images(image0_path, image1_path, visualize=True):\n",
    "    \"\"\"Perform OmniGlue matching between two images.\n",
    "    \n",
    "    Args:\n",
    "        image0_path: Path to the first image\n",
    "        image1_path: Path to the second image\n",
    "        visualize: Whether to visualize the matches\n",
    "        \n",
    "    Returns:\n",
    "        matches: Matched keypoints\n",
    "        visualization: Visualization of the matches if visualize=True\n",
    "    \"\"\"\n",
    "    # Load the images\n",
    "    image0 = np.array(Image.open(image0_path).convert('RGB'))\n",
    "    image1 = np.array(Image.open(image1_path).convert('RGB'))\n",
    "    \n",
    "    # Create the matcher\n",
    "    matcher = omniglue.OmniGlueMatcher()\n",
    "    \n",
    "    # Match the images\n",
    "    start_time = time.time()\n",
    "    result = matcher.match(image0, image1)\n",
    "    end_time = time.time()\n",
    "    \n",
    "    # Get match info\n",
    "    matches = result['matches']\n",
    "    num_matches = (matches > -1).sum()\n",
    "    match_confidence = result['matching_scores']\n",
    "    kpts0 = result['keypoints0']\n",
    "    kpts1 = result['keypoints1']\n",
    "    \n",
    "    print(f\"Number of matches: {num_matches}\")\n",
    "    print(f\"Time taken: {end_time - start_time:.2f} seconds\")\n",
    "    \n",
    "    if visualize:\n",
    "        # Create the visualization\n",
    "        visualization = utils.plot_matches(\n",
    "            image0, image1, kpts0, kpts1, matches, match_confidence,\n",
    "            plot_threshold=0.5, show_correspondence=True, show_keypoints=True)\n",
    "        \n",
    "        # Display the visualization\n",
    "        plt.figure(figsize=(15, 10))\n",
    "        plt.imshow(visualization)\n",
    "        plt.axis('off')\n",
    "        plt.title(f\"OmniGlue matches between {os.path.basename(image0_path)} and {os.path.basename(image1_path)}\")\n",
    "        plt.show()\n",
    "        \n",
    "        return matches, visualization\n",
    "    \n",
    "    return matches, None"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Test OmniGlue with Demo Images\n",
    "\n",
    "Now let's test OmniGlue with the demo images provided in the repository."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Test with demo images\n",
    "matches, visualization = match_images('./res/demo1.jpg', './res/demo2.jpg')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Test OmniGlue with Custom Images\n",
    "\n",
    "You can also test OmniGlue with your own images by uploading them to the `test_images` directory."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Upload your own images if you want to test with them\n",
    "# Then match them\n",
    "# matches, visualization = match_images('test_images/your_image1.jpg', 'test_images/your_image2.jpg')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Advanced: Test OmniGlue with Custom Parameters\n",
    "\n",
    "OmniGlue provides options to customize the matching process. Let's create a function that allows us to experiment with different parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "def match_images_custom(image0_path, image1_path, visualize=True, max_keypoints=1024, match_threshold=0.5):\n",
    "    \"\"\"Perform OmniGlue matching between two images with custom parameters.\n",
    "    \n",
    "    Args:\n",
    "        image0_path: Path to the first image\n",
    "        image1_path: Path to the second image\n",
    "        visualize: Whether to visualize the matches\n",
    "        max_keypoints: Maximum number of keypoints to detect\n",
    "        match_threshold: Threshold for confident matches\n",
    "        \n",
    "    Returns:\n",
    "        matches: Matched keypoints\n",
    "        visualization: Visualization of the matches if visualize=True\n",
    "    \"\"\"\n",
    "    # Load the images\n",
    "    image0 = np.array(Image.open(image0_path).convert('RGB'))\n",
    "    image1 = np.array(Image.open(image1_path).convert('RGB'))\n",
    "    \n",
    "    # Create the matcher with custom parameters\n",
    "    matcher = omniglue.OmniGlueMatcher(max_keypoints=max_keypoints)\n",
    "    \n",
    "    # Match the images\n",
    "    start_time = time.time()\n",
    "    result = matcher.match(image0, image1)\n",
    "    end_time = time.time()\n",
    "    \n",
    "    # Get match info\n",
    "    matches = result['matches']\n",
    "    match_confidence = result['matching_scores']\n",
    "    kpts0 = result['keypoints0']\n",
    "    kpts1 = result['keypoints1']\n",
    "    \n",
    "    # Count confident matches\n",
    "    confident_matches = (match_confidence > match_threshold).sum()\n",
    "    \n",
    "    print(f\"Number of keypoints in image 1: {len(kpts0)}\")\n",
    "    print(f\"Number of keypoints in image 2: {len(kpts1)}\")\n",
    "    print(f\"Total matches: {(matches > -1).sum()}\")\n",
    "    print(f\"Confident matches (score > {match_threshold}): {confident_matches}\")\n",
    "    print(f\"Time taken: {end_time - start_time:.2f} seconds\")\n",
    "    \n",
    "    if visualize:\n",
    "        # Create the visualization\n",
    "        visualization = utils.plot_matches(\n",
    "            image0, image1, kpts0, kpts1, matches, match_confidence,\n",
    "            plot_threshold=match_threshold, show_correspondence=True, show_keypoints=True)\n",
    "        \n",
    "        # Display the visualization\n",
    "        plt.figure(figsize=(15, 10))\n",
    "        plt.imshow(visualization)\n",
    "        plt.axis('off')\n",
    "        plt.title(f\"OmniGlue matches (threshold: {match_threshold})\")\n",
    "        plt.show()\n",
    "        \n",
    "        return matches, match_confidence, visualization\n",
    "    \n",
    "    return matches, match_confidence, None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Test with custom parameters\n",
    "matches, match_confidence, visualization = match_images_custom(\n",
    "    './res/demo1.jpg', './res/demo2.jpg', max_keypoints=2048, match_threshold=0.7)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Compare OmniGlue with Traditional Methods\n",
    "\n",
    "Let's compare OmniGlue with a traditional method like SIFT to see the difference in matching quality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "def match_images_sift(image0_path, image1_path, visualize=True, max_keypoints=1024):\n",
    "    \"\"\"Perform SIFT matching between two images.\n",
    "    \n",
    "    Args:\n",
    "        image0_path: Path to the first image\n",
    "        image1_path: Path to the second image\n",
    "        visualize: Whether to visualize the matches\n",
    "        max_keypoints: Maximum number of keypoints to detect\n",
    "        \n",
    "    Returns:\n",
    "        matches: Matched keypoints\n",
    "        visualization: Visualization of the matches if visualize=True\n",
    "    \"\"\"\n",
    "    # Load the images\n",
    "    image0 = cv2.imread(image0_path)\n",
    "    image1 = cv2.imread(image1_path)\n",
    "    \n",
    "    # Convert to grayscale\n",
    "    gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)\n",
    "    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)\n",
    "    \n",
    "    # Create SIFT detector\n",
    "    sift = cv2.SIFT_create(nfeatures=max_keypoints)\n",
    "    \n",
    "    # Detect keypoints and compute descriptors\n",
    "    start_time = time.time()\n",
    "    kp0, desc0 = sift.detectAndCompute(gray0, None)\n",
    "    kp1, desc1 = sift.detectAndCompute(gray1, None)\n",
    "    \n",
    "    # Match descriptors\n",
    "    bf = cv2.BFMatcher()\n",
    "    matches = bf.knnMatch(desc0, desc1, k=2)\n",
    "    end_time = time.time()\n",
    "    \n",
    "    # Apply ratio test\n",
    "    good_matches = []\n",
    "    for m, n in matches:\n",
    "        if m.distance < 0.75 * n.distance:\n",
    "            good_matches.append(m)\n",
    "    \n",
    "    print(f\"Number of keypoints in image 1: {len(kp0)}\")\n",
    "    print(f\"Number of keypoints in image 2: {len(kp1)}\")\n",
    "    print(f\"Number of matches: {len(good_matches)}\")\n",
    "    print(f\"Time taken: {end_time - start_time:.2f} seconds\")\n",
    "    \n",
    "    if visualize:\n",
    "        # Create the visualization\n",
    "        visualization = cv2.drawMatches(image0, kp0, image1, kp1, good_matches, None,\n",
    "                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)\n",
    "        \n",
    "        # Convert BGR to RGB for display\n",
    "        visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "        # Display the visualization\n",
    "        plt.figure(figsize=(15, 10))\n",
    "        plt.imshow(visualization)\n",
    "        plt.axis('off')\n",
    "        plt.title(f\"SIFT matches between {os.path.basename(image0_path)} and {os.path.basename(image1_path)}\")\n",
    "        plt.show()\n",
    "        \n",
    "        return good_matches, visualization\n",
    "    \n",
    "    return good_matches, None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Test with SIFT\n",
    "sift_matches, sift_visualization = match_images_sift('./res/demo1.jpg', './res/demo2.jpg', max_keypoints=1024)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Conclusion\n",
    "\n",
    "In this notebook, we've explored OmniGlue, a generalizable image feature matching library from Google Research that leverages foundation models to improve matching across different domains.\n",
    "\n",
    "We've:\n",
    "1. Set up the environment and installed OmniGlue\n",
    "2. Downloaded the required models\n",
    "3. Tested OmniGlue with demo images\n",
    "4. Created functions for customized matching\n",
    "5. Compared OmniGlue with traditional SIFT matching\n",
    "\n",
    "OmniGlue demonstrates how foundation models can improve traditional computer vision tasks by providing generalizable guidance for feature matching."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

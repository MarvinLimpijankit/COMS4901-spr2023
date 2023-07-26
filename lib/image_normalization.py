# import packages
import os
import cv2

"""
find_smallest_img_size: 

--> Method to find the smallest dimensions for 'raw' images in a dataset

--> Inputs: 
  - input_path, a string defining the path of the general folder containing the images (e.g. '../data') 
  - websites, a list of (NEWS SOURCE, REGION) tuples defining the subdirectories of the images (e.g. ('CNN', 'US') for path '../data/CNN/US') 

--> Outputs: 
  -  (min_x, min_y), a tuple with the smallest number of pixels in each direction of all images
"""
def find_smallest_img_size(input_path, websites):

    # initialize a list to keep track of image dimensions
    shapes = []

    # parse through all images, populating the list
    for website in websites:
        path = '{}/{}/{}'.format(input_path, website[1], website[0])
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename[-4:] == '.jpg':
                img = cv2.imread(path + '/' + filename)
                shapes.append(img.shape)

    # iterate through the dimensions, finding the global minimum
    min_x = float('inf')
    min_y = float('inf')

    for i in shapes:
        if i[0] < min_x:
            min_x = i[0]
        if i[1] < min_y:
            min_y = i[1]

    # print results to user
    print('The minimum width across all images is: {} pixels'.format(min_x))
    print('The minimum height across all images is: {} pixels'.format(min_y))

    return (min_x, min_y)

"""
normalize_imgs: 

--> Method to normalize all images to size x size dimensions saving them in 'output_path/normalized(SIZE, SIZE)'

--> Inputs: 
  - output_path, a string defining the general folder to the output location of the normalized images (e.g. '../output')
  - input_path, a string defining the general folder of the images (e.g. '../data') 
  - websites, a list of (NEWS SOURCE, REGION) tuples defining the subdirectories of the images (e.g. ('CNN', 'US') for path '../data/CNN/US')
  
--> Outputs: 
  -  
"""
def normalize_imgs(output_path, input_path, websites, size):

    # create the output path
    new_path = '{}/normalized{}'.format(output_path, size)
    os.mkdir(new_path)

    # iterate through sources, normalizing each image and saving them
    for website in websites:

        if not os.path.exists(new_path + '/{}'.format(website[1])):
            os.mkdir(new_path + '/{}'.format(website[1]))

        img_count = 0

        path = '{}/{}/{}'.format(input_path, website[1], website[0])
        new_website_path = new_path + '/{}/{}'.format(website[1], website[0])

        os.mkdir(new_website_path)

        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename[-4:] == '.jpg':
                img = cv2.imread(path + '/' + filename)
                img = cv2.resize(img, size)
                cv2.imwrite(new_website_path + '/img_{}.jpg'.format(img_count), img)
            img_count += 1

    print('Images have been normalized to size {} and saved at path "{}".'.format(size, new_path))


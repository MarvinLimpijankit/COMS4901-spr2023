# import packages
import os
import cv2
import colorsys
import numpy as np
import pandas as pd
from scipy import stats
from scipy import stats as st
import matplotlib.pyplot as plt
from scipy.spatial import distance
import matplotlib.patches as mpatches
from scipy.stats import kurtosis, skew

"""
extract_img_data: 

--> Method to extract pixel data from an image into a .csv file, where each row represents an image and each column represents a pixel in the specified color-space

--> Inputs: 
  - input_path, a string defining the general folder of the images (e.g. '../output/normalized(128, 128)') 
  - output_path, a string defining the general folder to the output location of the .csv (e.g. '../output')
  - websites, a list of (NEWS SOURCE, REGION) tuples defining the subdirectories of the images (e.g. ('CNN', 'US') for path '../input_path/CNN/US')
  - img_format, a string defining the color-space to be used ('rgb', 'hsv', or 'lab') 

--> Outputs: 
  -  returns a dataframe with the aggregated image data
"""
def extract_img_data(input_path, output_path, websites, img_format):

    # list to store output dataframe
    out = []

    # iterate through all images, converting the images into rows
    for website in websites:

        path = '{}/{}/{}'.format(input_path, website[1], website[0])

        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename[-4:] == '.jpg':
                img = cv2.imread(path + '/' + filename)

                if img_format == 'rgb':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img_format == 'hsv':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                elif img_format == 'lab':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                else:
                    raise Exception("Invalid color space, please choose from: rgb, hsv, lab")

                img = np.array(img)
                tmp = [website[1], website[0], filename] + [list(i) for i in list(img.reshape(16384, 3))]
                out.append(tmp)

    # convert the multidimensional list into a pandas dataframe
    out = pd.DataFrame(out)

    # save the dataframe as a .csv file
    out.to_csv('{}/img_data_{}.csv'.format(output_path, img_format), index=False, header=False)

    print('Extracted {} data to file {}/img_data_{}.csv'.format(img_format, output_path, img_format))

    # return the dataframe
    return out

"""
plt_3d_rgb:

--> Creates a 3D plot for an image where each point is a pixel, and each axis is a dimension (r, g, or b)

--> Inputs: 
  - source, a string defining the name of the image's news source (e.g. 'CNN') 
  - img_num, an integer defining the indexed number of the image (e.g. 3 for 'CNN/img_3') 
  - rgb_df, a dataframe containing the image's (rows) pixel (columns) information for all images in rgb space

--> Outputs: 
  -  
"""
def plt_3d_rgb(source, img_num, rgb_df):

    # locate the specified image
    rgb_vals = rgb_df[(rgb_df[1] == source) & (rgb_df[2] == 'img_{}.jpg'.format(img_num))].values[0][3:]
    rgb_vals = [i[1:-1].split(', ') for i in rgb_vals]
    rgb_vals = np.array(rgb_vals).astype(float)

    # for each pixel, gather the coordinates and color-map
    x, y, z, c = [], [], [], []

    for pix in rgb_vals:
        x.append(pix[0])
        y.append(pix[1])
        z.append(pix[2])
        c.append((pix[0] / 255, pix[1] / 255, pix[2] / 255))

    # plot the results
    fig = plt.figure()
    fig.tight_layout()
    plt.axis('off')
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c=c)
    plt.title('RGB Color Space Visualization for source:{} image number:{}'.format(source, img_num))
    plt.show()

"""
plt_3d_rgb_website_agg:

--> Creates a 3D plot showing the average pixel information of images by source, where each point represents one source and each axis is a dimension (r, g, or b)

--> Inputs: 
  - websites, a list of strings with source names (e.g. ['CNN', 'NBC', ...]) 
  - rgb_df, a dataframe containing the image's (rows) pixel (columns) information for all images in rgb space
  
--> Outputs: 
  -  
"""
def plt_3d_rgb_website_agg(websites, rgb_df):

    # create a dictionary to store news_source --> [average_r, average_g, average_b]
    avg_pixel = {}

    # iterate through all websites calculating the average pixel's r, g, and b values
    for website in websites:

        r_vals = []
        g_vals = []
        b_vals = []

        tmp_df = rgb_df[rgb_df[1] == website]

        for index, row in tmp_df[tmp_df[1] == website].iterrows():

            for i in range(3, 16387):
                pixel = row[i][1:-1].split(', ')

                r_vals.append(float(pixel[0]))
                g_vals.append(float(pixel[1]))
                b_vals.append(float(pixel[2]))

        avg_pixel[website] = [np.mean(r_vals), np.mean(b_vals), np.mean(g_vals)]

    # plot the results
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    plt.axis('off')
    ax = plt.axes(projection='3d')

    # create a legend for each news source
    for key in avg_pixel.keys():
        ax.scatter(avg_pixel[key][0], avg_pixel[key][1], avg_pixel[key][2], label=key)

    plt.title('RGB Color Space Visualization')
    plt.legend()
    plt.savefig('../figs/rgb/rgb_space_summary.png')
    plt.show()

"""
plt_channel_distributions:

--> Creates a histogram plot for each of the websites for their 1st, 2nd, and 3rd pixel channel

--> Inputs: 
  - websites, a list of strings with 3 source names (e.g. ['CNN', 'NBC', 'NYT']) 
  - df, a dataframe containing the image's (rows) pixel (columns) information for all images in a specified color space
  - fig_output, a string defining the path to save the visualization to
  - color_space, a string defining the color space (e.g. 'rgb', 'lab', or 'hsv')
  
--> Outputs: 
  -  
"""
def plt_channel_distributions(websites, df, fig_output, color_space):

    # dictionaries to store the distributions by source
    distributions_1 = {}
    distributions_2 = {}
    distributions_3 = {}

    # iterate through sources
    for website in websites:

        # lists to collect pixel values
        vals_1 = []
        vals_2 = []
        vals_3 = []

        # slice the dataframe
        tmp_df = df[df[1] == website]

        # iterate through each image
        for index, row in tmp_df[tmp_df[1] == website].iterrows():

            # specifically for a 128x128 image
            for i in range(3, 16387):

                pixel = row[i][1:-1].split(', ')

                # update values
                vals_1.append(float(pixel[0]))
                vals_2.append(float(pixel[1]))
                vals_3.append(float(pixel[2]))

        distributions_1[website] = vals_1
        distributions_2[website] = vals_2
        distributions_3[website] = vals_3

    fig, axes = plt.subplots(3, 3, figsize=(20, 10))

    for i, website in enumerate(websites):

        if color_space == 'rgb' or color_space == 'hsv':
            n_bins = [90, 128, 128]

        else:
            n_bins = [128, 128, 128]

        vals_arr = [distributions_1[website], distributions_2[website], distributions_3[website]]

        if color_space == 'rgb':
            titles = ['r Distribution', 'g Distribution', 'b Distribution']

        elif color_space == 'hsv':
            titles = ['Hue Distribution', 'Saturation Distribution', 'Values Distribution']

        elif color_space == 'lab':
            titles = ['l Distribution', 'a Distribution', 'b Distribution']

        else:
            raise Exception("color_space has to be rgb, hsv, or lab")

        for j in range(3):
            N, bins, patches = axes[i][j].hist(vals_arr[j], n_bins[j],
                                               weights=np.ones(len(vals_arr[j])) / len(vals_arr[j]))
            axes[i][j].set_title(website + ': ' + titles[j])

            # neat trick to color the 'hue' distribution for hsv
            if color_space == 'hsv' and j == 0:
                for k in range(len(patches)):
                    c = colorsys.hsv_to_rgb(bins[k] / 180, 1, 1)
                    patches[k].set_facecolor(c)

    # plot the figure
    plt.tight_layout()

    # save the plot
    plt.savefig(fig_output)

"""
plt_channel_distribution_by_source_3d:

--> Creates a 3d histogram plot for each of the websites for a specified channel

--> Inputs: 
  - title, a string with the general source title (e.g. 'US' or 'China')
  - websites, a list of strings with 3 source names from the same title (e.g. ['CNN', 'NBC', 'NYT']) 
  - colors, a list of 3 strings specifying the color to be used for each distribution (e.g. ['r', 'g', 'b'])
  - df, a dataframe containing the image's (rows) pixel (columns) information for all images in a specified color space
  - fig_output, a string defining the path to save the visualization to
  - channel, an integer denoting the channel to visualize (e.g. 1 for the second channel such as 'g' in rgb)

--> Outputs: 
  -  
"""
def plt_channel_distribution_by_source_3d(title, websites, colors, df, fig_output, channel):

    # store the distributions for each source in a dictionary
    distributions = {}

    # iterate through sources
    for website in websites:

        # store the pixel-level values
        vals = []

        tmp_df = df[df[1] == website]

        # for each image
        for index, row in tmp_df[tmp_df[1] == website].iterrows():

            # for each pixel
            for i in range(3, 16387):

                # store the information
                pixel = row[i][1:-1].split(', ')
                vals.append(float(pixel[channel]))

        # store the distribution
        distributions[website] = vals

    # === PLOTS THE 3D HISTOGRAM USING THE DATA AND SPECIFIED VISUAL PARAMETERS ===

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    nbins = 128

    for c, z, website in zip(colors, [20, 10, 0], websites):
        ys = distributions[website]
        hist, bins = np.histogram(ys, bins=nbins,
                                  weights=np.ones(len(distributions[website])) / len(distributions[website]))
        xs = (bins[:-1] + bins[1:]) / 2

        ax.bar(xs, hist, zs=z, zdir='y', color=c, ec=c, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    patch_1 = mpatches.Patch(color=colors[0], label=websites[0])
    patch_2 = mpatches.Patch(color=colors[1], label=websites[1])
    patch_3 = mpatches.Patch(color=colors[2], label=websites[2])

    ax.set_zlim(0, 0.01)
    plt.legend(handles=[patch_1, patch_2, patch_3])
    plt.title(title)

    # save the plot
    plt.savefig(fig_output)

    # display the plot
    plt.show()

"""
get_distributions:

--> For a list of websites, gather and return their pixel-level channel distributions as a dictionary pointing to arrays

--> Inputs: 
  - websites, a list of strings with source names (e.g. ['CNN', 'NBC', ...]) 
  - df, a dataframe containing the image's (rows) pixel (columns) information for all images in a specific color space

--> Outputs: 
  -  distributions_1, distributions_2, distributions_3, dictionaries with source names as keys and distribution arrays as values
"""
def get_distributions(websites, df):

    # dictionaries to store the distributions for each channel
    distributions_1 = {}
    distributions_2 = {}
    distributions_3 = {}

    # iterate through websites
    for website in websites:

        # arrays to store the pixel-level information
        vals_1 = []
        vals_2 = []
        vals_3 = []

        # slice the dataframe onto a specific website
        tmp_df = df[df[1] == website]

        # for each image
        for index, row in tmp_df[tmp_df[1] == website].iterrows():

            # for each pixel
            for i in range(3, 16387):
                pixel = row[i][1:-1].split(', ')

                # collect pixel-level values
                vals_1.append(float(pixel[0]))
                vals_2.append(float(pixel[1]))
                vals_3.append(float(pixel[2]))

        # update the dictionaries
        distributions_1[website] = vals_1
        distributions_2[website] = vals_2
        distributions_3[website] = vals_3

    # return the results
    return distributions_1, distributions_2, distributions_3

"""
get_jenson_shannon_distances:

--> Calculates the pairwise distances between distributions using jenson_shannon entropy

--> Inputs: 
  - websites, a list of strings containing source names (e.g. ['CNN', 'NBC', ...]) 
  - distributions_dict, dictionary with source as a key and a distribution (array) as a value for the websites above for a specific channel
  
--> Outputs: 
  - jenson_shannon_dist, a dictionary with tuples (pairs of sources) as the key and distances as a value
  - arr, a 2D array containing the jenson shannon distances as a pairwise matrix
"""
def get_jenson_shannon_distances(websites, distributions_dict):

    # variables to store output
    jensen_shannon_dist = {}
    arr = []

    # iterate through pairs of websites
    for website1 in websites:
        row = []

        for website2 in websites:

            # gather distributions
            hist1, bins1 = np.histogram(distributions_dict[website1], bins=128,
                                        weights=np.ones(len(distributions_dict[website1])) / len(
                                            distributions_dict[website1]))
            hist2, bins2 = np.histogram(distributions_dict[website2], bins=128,
                                        weights=np.ones(len(distributions_dict[website2])) / len(
                                            distributions_dict[website2]))

            # compute the distance
            jensen_shannon_dist[(website1, website2)] = distance.jensenshannon(hist1, hist2)
            row.append(jensen_shannon_dist[(website1, website2)])

        # update the results
        arr.append(row)

    # return the results
    return jensen_shannon_dist, arr

"""
get_ks_test_p_vals:

--> Calculates the pairwise p_values between distributions using a KS test

--> Inputs: 
  - websites, a list of strings containing source names (e.g. ['CNN', 'NBC', ...]) 
  - distributions_dict, dictionary with source as a key and a distribution (array) as a value for the websites above for a specific channel

--> Outputs: 
  - ks_test_p_val, a dictionary with tuples (pairs of sources) as the key and p_values as a value
  - arr, a 2D array containing the p_values as a pairwise matrix
"""
def get_ks_test_p_vals(websites, distributions_dict):

    # variables to store output
    ks_test_p_val = {}
    arr = []

    # iterate through pairs of websites
    for website1 in websites:
        row = []

        for website2 in websites:

            # gather distributions
            hist1, bins1 = np.histogram(distributions_dict[website1], bins=128,
                                        weights=np.ones(len(distributions_dict[website1])) / len(
                                            distributions_dict[website1]))
            hist2, bins2 = np.histogram(distributions_dict[website2], bins=128,
                                        weights=np.ones(len(distributions_dict[website2])) / len(
                                            distributions_dict[website2]))

            # compute the distance
            ks_test_p_val[(website1, website2)] = stats.kstest(hist1, hist2).pvalue
            row.append(ks_test_p_val[(website1, website2)])

        # update the results
        arr.append(row)

    # return the results
    return ks_test_p_val, arr

"""
get_ks_test_p_vals:

--> Given a dictionary of sources --> distributions, return a summary report of the standard metrics of the distribution

--> Inputs: 
  - distributions_dict, a dictionary with source names as keys and distributions (array of values) as a value
  - print_bool, a boolean that determines whether to print the statistics or not
  
--> Outputs: 
  - output, a dictionary with sources names as keys and corresponding distribution metrics as values
"""

def get_standard_metrics(distributions_dict):

    print("SUMMARY: \n\n")

    # iterate through sources
    for website in distributions_dict.keys():

        vals = distributions_dict[website]

        # website name
        print("===== {} =====".format(website))

        # mean
        print("--> mean: {}".format(np.mean(vals)))

        # median
        print("--> median: {}".format(np.median(vals)))

        # mode
        print("--> mode: {}".format(st.mode(vals).mode[0]))

        # max
        print("--> max: {}".format(max(vals)))

        # min
        print("--> min: {}".format(min(vals)))

        # skew
        print("--> skew: {}".format(skew(vals)))

        # kurtosis
        print("--> kurtosis: {}\n".format(kurtosis(vals)))

    return

"""
get_standard_metrics_df:

--> Given a dataframe with images as rows and pixel information as columns and the image space, return a dataframe with the channel distribution metrics on an image-level

--> Inputs: 
  - df, a dataframe with pixel information on an image-level (see extract_image_data)
  - space, a string defining a specified image space ie. 'lab', 'hsv', etc. 

--> Outputs: 
  - output, a DataFrame with channel distribution metrics on an image-level
"""
def get_standard_metrics_df(df, space):

    # set up the headers for the output DataFrame
    metrics = ['mean', 'median', 'mode', 'max', 'min', 'skew', 'kurt']
    headers = ['region', 'source', 'image'] + [space[0] + '_' + i for i in metrics] + [space[1] + '_' + i for i in metrics] + [space[2] + '_' + i for i in metrics]

    # initialize output
    out = []

    # iterate through each image
    for i in df.iterrows():

        # gather metadata
        region = i[1][0]
        source = i[1][1]
        img_name = i[1][2]

        # lists to stores pixel values at the channel level
        chan_1 = []
        chan_2 = []
        chan_3 = []

        # iterate through pixels, populating the data
        for pix in i[1][3:]:
            chan_1.append(int(pix[1:-1].split(", ")[0]))
            chan_2.append(int(pix[1:-1].split(", ")[1]))
            chan_3.append(int(pix[1:-1].split(", ")[2]))

        # calculate standard metrics on an image level
        chan_1_metrics = [np.mean(chan_1), np.median(chan_1), st.mode(chan_1).mode[0], max(chan_1), min(chan_1),
                          skew(chan_1), kurtosis(chan_1)]
        chan_2_metrics = [np.mean(chan_2), np.median(chan_2), st.mode(chan_2).mode[0], max(chan_2), min(chan_2),
                          skew(chan_2), kurtosis(chan_2)]
        chan_3_metrics = [np.mean(chan_3), np.median(chan_3), st.mode(chan_3).mode[0], max(chan_3), min(chan_3),
                          skew(chan_3), kurtosis(chan_3)]

        # append to the output
        out.append([region, source, img_name] + chan_1_metrics + chan_2_metrics + chan_3_metrics)

    # convert output to a DataFrame
    out = pd.DataFrame(out, columns=headers)

    # return
    return out
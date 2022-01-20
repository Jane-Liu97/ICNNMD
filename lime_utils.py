

import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime.wrappers.scikit_image import SegmentationAlgorithm
from lime import lime_image
import numpy as np


def cnn_lime(model, X, y, k):

    

#     %load_ext autoreload
#     %autoreload 2
    

    explainer = lime_image.LimeImageExplainer(verbose = False)
    segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)
    # =============================================================================

    # =============================================================================
    
    n_features = 100
    n_samples = 1000

    importance_pic = []
    size = len(X[0])
    n_lime = int(len(y)/10)
    for i in range(size):
        line = []
        for j in range(size):
            line.append(0)
        importance_pic.append(line)
        
#     for i in range(n_lime):
#         print(i)
#         explanation = explainer.explain_instance(X[i*10].astype('float64'), 
#                                                  classifier_fn = model[0].predict_proba,  
#                                                  top_labels=2, hide_color=0, num_samples=n_samples, segmentation_fn=segmenter)
#         temp, mask = explanation.get_image_and_mask(y[i*10], positive_only=True, num_features=n_features, hide_rest=False)
#         importance_pic = np.array(importance_pic) + mask
        
    for j in range(k):
        for i in range(n_lime):
            print(i)
            explanation = explainer.explain_instance(X[i*10].astype('float64'), 
                                                     classifier_fn = model[j].predict_proba,  
                                                     top_labels=2, hide_color=0, num_samples=n_samples, segmentation_fn=segmenter)
            temp, mask = explanation.get_image_and_mask(y[i*10], positive_only=True, num_features=n_features, hide_rest=False)
            
            importance_pic = np.array(importance_pic) + np.array(mask)
#             if(sum(sum(mask)))!=0:
#                 mask = [x/(sum(sum(mask))) for x in mask]
#                 importance_pic = np.array(importance_pic) + np.array(mask)
    
    
    # =============================================================================

    return importance_pic, explainer, segmenter


def lime_show(explainer, segmenter, model, X, y, k, pixel_map):
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    
    n_features = 100
    n_samples = 1000
    
#     explanation = explainer.explain_instance(X, classifier_fn = model[0].predict_proba, 
#                                              top_labels=2, hide_color=0, num_samples=n_samples, segmentation_fn=segmenter)
#     temp, mask = explanation.get_image_and_mask(y, positive_only=True, num_features=n_features, hide_rest=False)
    
    explanation = explainer.explain_instance(X, classifier_fn = model[0].predict_proba, 
                                             top_labels=2, hide_color=0, num_samples=n_samples, segmentation_fn=segmenter)
    temp, mask = explanation.get_image_and_mask(y, positive_only=True, num_features=n_features, hide_rest=False)

    masks = mask
    for i in range(k):
        if i!=0:
            explanation = explainer.explain_instance(X, classifier_fn = model[i].predict_proba, 
                                                     top_labels=2, hide_color=0, num_samples=n_samples, segmentation_fn=segmenter)
            temp, mask = explanation.get_image_and_mask(y, positive_only=True, num_features=n_features, hide_rest=False)

            masks = masks + mask
            
#     # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
#     fig, (ax1) = plt.subplots(1, figsize = (4,4))
#     temp = np.array(pixel_map)
#     ax1.imshow(label2rgb(masks,temp, bg_label = 0), interpolation = 'nearest')
#     ax1.set_title('Positive Regions for {}'.format(y))
    
    
    temp = np.array(pixel_map)
    signs = []
    for i in range(len(masks)):
        line = []
        for j in range(len(masks[0])):
            if masks[i][j]==0:
                line.append([255,255,255])
            elif masks[i][j]==1:
                line.append([211,211,211])
            elif masks[i][j]==2:
                line.append([192,192,192])
            elif masks[i][j]==3:
                line.append([128,128,128])
            elif masks[i][j]==4:
                line.append([105,105,105])
            else:
                line.append([0,0,0])
        signs.append(line)
        
    signs = np.array(signs)

    import cv2
    img_a = signs.astype("uint8")
    img_b = temp.astype("uint8")

    lucency = 0.5
    img_a=cv2.resize(img_a,(img_b.shape[1],img_b.shape[0]))

    img_c=lucency*img_b+(1-lucency)*img_a
    img_c=img_c.astype(np.uint8)
    img_c=np.clip(img_c,0,255)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
    ax1.imshow(img_a)
    ax2.imshow(img_c)
    fig.show()


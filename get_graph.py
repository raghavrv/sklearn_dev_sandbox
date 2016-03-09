import matplotlib.pyplot as plt
import pygraphviz as pgv
import networkx as nx
import pygraphviz
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from value_dropper import drop_values

from sklearn.tree import *
from StringIO import StringIO
from io import BytesIO
import numpy as np

def get_graph(dtc, n_classes, feat_names=None, size=[7, 7], max_depth=10):
    dot_file = StringIO()
    image_file = BytesIO()

    # Get the dot graph of our decision tree
    export_graphviz(dtc, out_file=dot_file, feature_names=feat_names,
                    rounded=True, filled=True,
                    special_characters=True,
                    class_names=map(str, range(1, n_classes+1)),
                    max_depth=max_depth)
    dot_file.seek(0)

    # Convert this dot graph into an image
    g = pygraphviz.AGraph(dot_file.read())
    g.layout('dot')
    # g.draw doesn't work when the image object doesn't have a name (with a proper extension)
    image_file.name = "image.png"
    image_file.seek(0)
    g.draw(path=image_file)
    image_file.seek(0)

    # Plot it
    plt.figure().set_size_inches(*size)
    plt.axis('off')
    plt.imshow(img.imread(fname=image_file))
    plt.show()

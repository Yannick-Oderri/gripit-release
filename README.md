Application Name: GripIt

**Description**

Provided the depth map of an arbitrary scene, GripIt can extract the geometric edges of objects in said scene and further calculate the optimal approach vector for a 2-finger pinch-based robotic grippers. This information is presented in an interactive 3D point cloud view. Furthermore, GripIt provides editable parameters which governs particular features of the 2D and 3D scene. As a high-level overview, GripIt relies on machine vision algorithms to define the edges within a depth map. These edges are then paired and a normal vector calculated based on the underlying surface&#39;s depth map representation.

**Instruction**

**Application Dependencies.**

Currently, GripIt was built using Anaconda&#39;s build environment. GritIp also relies on the following packages:

-
  - OpenCV3                3.1.0
  - Matplotlib                2.1.0
  - Numpy                        1.13.3
  - PyQt                        5.6.0
  - PqQtGraph                0.10.0
  - Scipy                        0.19.1
  - Scikit-image                0.13.0

**Launching GripIt:**

Currently, GripIt must be launched from a terminal and depends on application arguments to load a scene.

Arguments:

- -m database[blend,real]
  - Selects a database where a set of scenes are stored. The &quot;Blend,&quot; stores synthetic data produced by blender while &quot;Real&quot; hosts an array of real images.
- -n imageNumber
  - Scenes stored in the database are selected by their numeric index.

For instance, to load the second scene from the database, &quot;real&quot;, the following commands must be used.

        $/ Python ./application.py -m real  -n 2

Scene Parameters:

![image](https://user-images.githubusercontent.com/847804/47836606-63380180-dd7f-11e8-88ac-6b650e2bd20c.png)


GripIt incorporates a set of parameters which may be used to alter the edge detection and point-cloud representation of the scene.

        Parameters:

- Auto-Canny Sigma
  - Controls the sensitivity for edge detection algorithms that are used. Lower value may exclude some edges, while a higher value may present noise. The default value of 33% is a statistical recommendation.
- Segmentation        Tolerance
  - Influences at what angle an edge may be divided.
- Minimum Paring Length Ratio
- Edge Pair Min Distance
  - Sets the minimum distance that an edge-pair has to be in order to be processed
-  Edge Pair Max Distance
  - Sets the maximum distance that an edge-pair has to be in order to be processed
- Edge Pair Angle
  - The maximum angle between 2 edge pair vectors

**Processing a Scene:**

On loading a scene, GripIt will launch the Base view as show in figure 1. Here the program parameters are edited, a region of interest is established under the crop rectangle of the image, and the scene processed by clicking the process button.

![image](https://user-images.githubusercontent.com/847804/47836771-dccfef80-dd7f-11e8-8450-94cba4f4ef22.png)

After a scene has been processed, a set of views will be added to the base window as tabs. These views present the calculated edge-pairs and approach vector as a 2D image and a 3D point-cloud. At any time the parameters could be re-edited and the scene re-updated with leaving the application.



Edge-Pair View:

![image](https://user-images.githubusercontent.com/847804/47836814-0ab53400-dd80-11e8-8d94-86517731bb55.png)


The first of these tabs, EdgePairs, displays the edge-pairs located in the cropped scene.  These edge pairs are color coded, and given numeric names. The underlying points defining an edge could also be viewed by selecting &quot;Display Edge Points.&quot; By pressing the left or right keys, a correspond 2d depth map image will be presented in the image view. To view the approach vector of an edge-pair, an edge-pair must be selected from the drop-down menu. Clicking &quot;process face&quot; generates an approach vector for the selected edge-pair. This vector could be viewed by switching to the Point-Cloud tab.

![image](https://user-images.githubusercontent.com/847804/47836838-215b8b00-dd80-11e8-8a90-a13437607edc.png)

In the point cloud tab, GripIt presents the scene in a 3D point cloud. The edge select tool of the EdgePair tab is synchronized with the edges that are shown in the PointCloud view. When an edge is processed the calculated approach vector is represented by a 2-finger gripper. This gripper dynamically resizes to grasp the selected edges. The scene presented in the Point-Cloud View could be panned and rotated as needed.

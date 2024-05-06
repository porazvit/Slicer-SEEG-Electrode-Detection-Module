import logging
import os
from typing import Annotated, Optional
from __main__ import vtk, qt, ctk, slicer


import vtk
import numpy as np
from skimage import measure
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from scipy.ndimage import uniform_filter
import re

import sys
sys.path.append('C:/Users/vitpo/OneDrive/Plocha/skola/semprojekt/slicer/project/beta1/HDBrainExtractionTool.py')
from HDBrainExtractionTool import *
import slicer
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# beta1
#

class beta1(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "beta1"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Vit Porazil (CTU)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#beta1">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # beta11
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='beta1',
        sampleName='beta11',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'beta11.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='beta11.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='beta11'
    )

    # beta12
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='beta1',
        sampleName='beta12',
        thumbnailFileName=os.path.join(iconsPath, 'beta12.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='beta12.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='beta12'
    )


#
# beta1ParameterNode
#

@parameterNodeWrapper
class beta1ParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# beta1Widget
#

class beta1Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    
    def __init__(self, parent = None):
        if not parent:
            self.parent = slicer.qMRMLWidget()
            self.parent.setLayout(qt.QVBoxLayout())
            self.parent.setMRMLScene(slicer.mrmlScene)
        else:
            self.parent = parent
            self.layout = self.parent.layout()

        self.logic_instance = beta1Logic()
        self.HDElogic_instance = HDBrainExtractionToolLogic()
        if not parent:
            self.setup()
            self.parent.show()

    def setup(self):
        # Collapsible button TOOLS
        self.ToolsCollapsibleButton = ctk.ctkCollapsibleButton()
        self.ToolsCollapsibleButton.text = "Tools"
        self.layout.addWidget(self.ToolsCollapsibleButton)

        # Layout within the collapsible button
        self.ToolsFormLayout = qt.QFormLayout(self.ToolsCollapsibleButton)


        restart = qt.QPushButton("Restart")
        restart.toolTip = "Restart Slicer"
        restart.connect('clicked(bool)', slicer.util.restart)
        self.ToolsFormLayout.addWidget(restart)

        # Reload button
        # reloadButton = qt.QPushButton("Reload Model")
        # reloadButton.toolTip = "Reload Model"
        # reloadButton.connect('clicked(bool)', slicer.util.reload("beta1"))
        # self.ToolsFormLayout.addWidget(reloadButton)

        #save = qt.QPushButton("Save")
        #save.toolTip = "Display"
        #save.connect('clicked(bool)', slicer.util.saveNode)
        #self.ToolsFormLayout.addWidget(save)

        exit = qt.QPushButton("Exit")
        exit.toolTip = "Exit Display"
        exit.connect('clicked(bool)', slicer.util.exit)
        self.ToolsFormLayout.addWidget(exit)



        # Collapsible button INPUTS
        self.InputsCollapsibleButton = ctk.ctkCollapsibleButton()
        self.InputsCollapsibleButton.text = "Inputs and Display"
        self.layout.addWidget(self.InputsCollapsibleButton)

        # Layout within the collapsible button
        self.InputsFormLayout = qt.QFormLayout(self.InputsCollapsibleButton)

        # create volume MRI
        self.inputFrame = qt.QFrame(self.InputsCollapsibleButton)
        self.inputFrame.setLayout(qt.QHBoxLayout())
        self.InputsFormLayout.addWidget(self.inputFrame)
        self.inputMRI = qt.QLabel("Input MRI:", self.inputFrame)
        self.inputFrame.layout().addWidget(self.inputMRI)
        self.inputMRI = slicer.qMRMLNodeComboBox(self.inputFrame)
        self.inputMRI.nodeTypes = (("vtkMRMLScalarVolumeNode"), "")
        self.inputMRI.addEnabled = True
        self.inputMRI.removeEnabled = True
        self.inputMRI.noneEnabled = False
        self.inputMRI.showHidden = False
        self.inputMRI.showChildNodeTypes = True
        self.inputMRI.setMRMLScene(slicer.mrmlScene)
        #self.inputMRI.currentText = '*FIXED*'
        self.inputFrame.layout().addWidget(self.inputMRI)

        # create volume CT
        self.inputFrameCT = qt.QFrame(self.InputsCollapsibleButton)
        self.inputFrameCT.setLayout(qt.QHBoxLayout())
        self.InputsFormLayout.addWidget(self.inputFrameCT)
        self.inputCT = qt.QLabel("Input CT:", self.inputFrameCT)
        self.inputFrameCT.layout().addWidget(self.inputCT)
        self.inputCT = slicer.qMRMLNodeComboBox(self.inputFrameCT)
        self.inputCT.nodeTypes = (("vtkMRMLScalarVolumeNode"), "")
        self.inputCT.addEnabled = True
        self.inputCT.removeEnabled = True
        self.inputCT.noneEnabled = False
        self.inputCT.showHidden = False
        self.inputCT.showChildNodeTypes = True
        self.inputCT.setMRMLScene(slicer.mrmlScene)

        self.inputFrameCT.layout().addWidget(self.inputCT)

        # Create fiducial list
        self.inputFrameFiducial = qt.QFrame(self.InputsCollapsibleButton)
        self.inputFrameFiducial.setLayout(qt.QHBoxLayout())
        self.InputsFormLayout.addWidget(self.inputFrameFiducial)
        self.inputFiducialLabel = qt.QLabel("Input Fiducial List:", self.inputFrameFiducial)
        self.inputFrameFiducial.layout().addWidget(self.inputFiducialLabel)
        self.inputFiducial = slicer.qMRMLNodeComboBox(self.inputFrameFiducial)
        self.inputFiducial.nodeTypes = (("vtkMRMLMarkupsFiducialNode"), "")
        self.inputFiducial.addEnabled = True
        self.inputFiducial.removeEnabled = True
        self.inputFiducial.noneEnabled = False
        self.inputFiducial.showHidden = False
        self.inputFiducial.showChildNodeTypes = True
        self.inputFiducial.setMRMLScene(slicer.mrmlScene)
        self.inputFrameFiducial.layout().addWidget(self.inputFiducial)

        # Create mask volume
        self.inputFrameMask = qt.QFrame(self.InputsCollapsibleButton)
        self.inputFrameMask.setLayout(qt.QHBoxLayout())
        self.InputsFormLayout.addWidget(self.inputFrameMask)
        self.inputMaskLabel = qt.QLabel("Input Mask Volume:", self.inputFrameMask)
        self.inputFrameMask.layout().addWidget(self.inputMaskLabel)
        self.inputMask = slicer.qMRMLNodeComboBox(self.inputFrameMask)
        self.inputMask.nodeTypes = (("vtkMRMLScalarVolumeNode"), "")
        self.inputMask.addEnabled = True
        self.inputMask.removeEnabled = True
        self.inputMask.noneEnabled = False
        self.inputMask.showHidden = False
        self.inputMask.showChildNodeTypes = True
        self.inputMask.setMRMLScene(slicer.mrmlScene)
        self.inputFrameMask.layout().addWidget(self.inputMask)


        # DISPLAY the current CT
        DisplayCT = qt.QPushButton("Display CT")
        DisplayCT.toolTip = "Display"
        DisplayCT.connect('clicked(bool)', self.displayCT)
        self.InputsFormLayout.addWidget(DisplayCT)

        #DISPLAY the current MRI
        DisplayMRI = qt.QPushButton("Display MRI")
        DisplayMRI.toolTip = "Display"
        DisplayMRI.connect('clicked(bool)', self.displayMRI)
        self.InputsFormLayout.addWidget(DisplayMRI)

        #DISPLAY the electrodes
        DisplayElectrodes = qt.QPushButton("Display Electrodes")
        DisplayElectrodes.toolTip = "Display 3D Render"
        DisplayElectrodes.connect('clicked(bool)', self.displayElectrodes)
        self.InputsFormLayout.addWidget(DisplayElectrodes)      

        #self.InputsFormLayout.addWidget(slicer.modules.VolumeRendering.widgetRepresentation())

        # Collapsible button REGISTER
        self.RegisterCollapsibleButton = ctk.ctkCollapsibleButton()
        self.RegisterCollapsibleButton.text = "Registration"
        self.RegisterCollapsibleButton.setChecked(False)  # Set to collapsed by default
        self.layout.addWidget(self.RegisterCollapsibleButton)

        self.RegisterFormLayout = qt.QFormLayout(self.RegisterCollapsibleButton)
        self.RegisterFormLayout.addWidget(slicer.modules.brainsfit.widgetRepresentation())

         # Apply Default button
        self.applyDefaultButton = qt.QPushButton("Apply Default")
        self.applyDefaultButton.toolTip = "Apply default registration"
        self.applyDefaultButton.connect('clicked(bool)', self.applyDefaultRegistration)
        self.RegisterFormLayout.addWidget(self.applyDefaultButton)

        #Segmantation collapsbile button
        self.SegmentationButton = ctk.ctkCollapsibleButton()
        self.SegmentationButton.text = "Segmantation"
        self.SegmentationButton.setChecked(False)  # Set to collapsed by default
        self.layout.addWidget(self.SegmentationButton)

        # Layout within the collapsible button
        self.SegmentationButtonLayout = qt.QFormLayout(self.SegmentationButton)

        #Add detect segmentation buton
        self.segmentationButton = qt.QPushButton("Segment Volume")
        self.segmentationButton.toolTip = "Run the HDBrainExtraction algorithm"
        self.segmentationButton.connect('clicked(bool)', self.onSegmentationButtonClicked)
        self.SegmentationButtonLayout.addWidget(self.segmentationButton)

        #Add modify mask button
        self.modifyMaskButton = qt.QPushButton("Create mask")
        self.modifyMaskButton.toolTip = "Masks out everything electrodes"
        self.modifyMaskButton.connect('clicked(bool)',self.getROIFromFiducial)
        self.SegmentationButtonLayout.addWidget(self.modifyMaskButton)

        #Add display mask button
        DisplayMask = qt.QPushButton("Display Mask")
        DisplayMask.toolTip = "Display 3D Render"
        DisplayMask.connect('clicked(bool)', self.displayMask)
        self.SegmentationButtonLayout.addWidget(DisplayMask)  


        # Collapsible button for adjusting electrodes
        self.AdjustElectrodesCollapsibleButton = ctk.ctkCollapsibleButton()
        self.AdjustElectrodesCollapsibleButton.text = "Adjust Electrodes"
        self.AdjustElectrodesCollapsibleButton.setChecked(False)  # Set to collapsed by default
        self.layout.addWidget(self.AdjustElectrodesCollapsibleButton)

        # Layout within the collapsible button
        self.AdjustElectrodesLayout = qt.QFormLayout(self.AdjustElectrodesCollapsibleButton)

        # Combo box for electrode selection
        self.adjustFiducialLabel = qt.QLabel("Select an electrode:")
        self.adjustFiducial = slicer.qMRMLNodeComboBox()
        self.adjustFiducial.nodeTypes = (("vtkMRMLMarkupsFiducialNode"), "")
        self.adjustFiducial.addEnabled = True
        self.adjustFiducial.removeEnabled = True
        self.adjustFiducial.noneEnabled = False
        self.adjustFiducial.showHidden = False
        self.adjustFiducial.showChildNodeTypes = True
        self.adjustFiducial.setMRMLScene(slicer.mrmlScene)
        self.AdjustElectrodesLayout.addRow(self.adjustFiducialLabel, self.adjustFiducial)

        # Label for shift amount
        self.shiftAmountLabel = qt.QLabel("Shift Amount:")
        self.shiftAmountLineEdit = qt.QLineEdit()
        self.AdjustElectrodesLayout.addRow(self.shiftAmountLabel, self.shiftAmountLineEdit)

        # Button for shifting points
        self.shiftPointsButton = qt.QPushButton("Shift Points")
        self.shiftPointsButton.toolTip = "Shift selected electrode points"
        self.AdjustElectrodesLayout.addWidget(self.shiftPointsButton)

        # Connect button click to shifting function
        self.shiftPointsButton.connect('clicked(bool)',self.onShiftPointsButtonClicked)
        #Add combine lists button
        self.combineListsButton = qt.QPushButton("Combine lists")
        self.combineListsButton.toolTip = "Combines all the electrode points lists into one single list"
        self.combineListsButton.connect('clicked(bool)', self.onCombineListsClicked)
        self.AdjustElectrodesLayout.addWidget(self.combineListsButton)

        # #Add detect reslice button
        # self.resliceImageButton = qt.QPushButton("Reslice")
        # self.resliceImageButton.toolTip = "reslice images"
        # self.resliceImageButton.connect('clicked(bool)',self.resliceImages)
        # self.layout.addWidget(self.resliceImageButton)

        #Add detect electrodes button
        self.detectElectrodesButton = qt.QPushButton("Detect Electrodes")
        self.detectElectrodesButton.toolTip = "Run the detection algorithm"
        self.detectElectrodesButton.connect('clicked(bool)',self.detectElectrodes)
        self.detectElectrodesButton.setStyleSheet("QPushButton { font-weight: bold; }")
        self.layout.addWidget(self.detectElectrodesButton)


        

    def onShiftPointsButtonClicked(self):
        selected_node = self.adjustFiducial.currentNode()
        if selected_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Error", "Please select an electrode.")
            return

        shift_amount = self.shiftAmountLineEdit.text
        self.fiducialData, newListNode = self.logic_instance.shiftPoints(selected_node, shift_amount, self.inputCT.currentNode(), self.fiducialData)
        self.adjustFiducial.setCurrentNode(newListNode)

    def onSegmentationButtonClicked(self):
        reslicedMRINode = self.resliceImages()
        outputVolume = slicer.vtkMRMLScalarVolumeNode()
        outputVolume.SetName("MaskedMRI")
        slicer.mrmlScene.AddNode(outputVolume)

        self.HDElogic_instance.process(reslicedMRINode, outputVolume, None, "auto" )
        self.inputMask.setCurrentNode(outputVolume)
        slicer.mrmlScene.RemoveNode(reslicedMRINode)

    def onCombineListsClicked(self):
        self.logic_instance.combineFiducialLists(self.fiducialData,  self.inputCT.currentNode(), self.electrode_names)
    

        
    def applyDefaultRegistration(self):
        #select MRI as fixed volume, CT as moving
        fixedVolume = self.inputMRI.currentNode()
        movingVolume = self.inputCT.currentNode()
        import time
        startTime = time.time()
        logging.info('Processing started')

        if fixedVolume is None or movingVolume is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Registration Error", "Please select both fixed and moving volumes.")
            return
    
        qt.QMessageBox.warning(slicer.util.mainWindow(), "Starting Registration", "Warning: This will take few minutes!")
        # Set CLI node parameters
        parameters = {
            'fixedVolume': fixedVolume.GetID(),
            'movingVolume': movingVolume.GetID(),
            'samplingPercentage': 0.01,
            'splineGridSize': [14, 10, 12],
            'initializeTransformMode': 'useMomentsAlign',
            'useRigid': True,
            'maskProcessingMode': 'NOMASK',
            'outputVolumePixelType': 'float',
            'backgroundFillValue': 0,
            'interpolationMode': 'Linear',
            'numberOfIterations': 1500,
            'maximumStepLength': 0.05,
            'minimumStepLength': 0.001,
            'relaxationFactor': 0.5,
            'translationScale': 1000,
            'reproportionScale': 1,
            'skewScale': 1,
            'maxBSplineDisplacement': 0,
            'numberOfThreads': -1,
            'costFunctionConvergenceFactor': 2e+13,
            'projectedGradientTolerance': 1e-05,
            'maximumNumberOfEvaluations': 900,
            'maximumNumberOfCorrections': 25,
            'metricSamplingStrategy': 'Random',
            'debugLevel': 0,
            'failureExitCode': -1,
            'outputVolume': movingVolume.GetID()  
        }


        # Register using CLI brainsfit module
        cliNode = slicer.cli.run(slicer.modules.brainsfit,None, parameters, wait_for_completion=True)
        
        # Clean up the CLI node
        slicer.mrmlScene.RemoveNode(cliNode)
        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
        registeredCTNode = slicer.util.getNode(parameters['outputVolume'])
        self.inputCT.setCurrentNodeID(registeredCTNode.GetID())
        
        qt.QMessageBox.information(slicer.util.mainWindow(), 'Registration Complete', 'Default registration applied successfully.')


    
    def displayCT(self):
        ct = self.inputCT.currentNode()
        slicer.util.setSliceViewerLayers(background=ct)

    def displayMRI(self):
        mri = self.inputMRI.currentNode()
        slicer.util.setSliceViewerLayers(background=mri)


    def displayElectrodes(self):
        ct = self.inputCT.currentNode()
        
        # Prepare node for output
        thresholdedVolume = slicer.vtkMRMLScalarVolumeNode()
        thresholdedVolume.SetName("ThresholdedVolume")
        slicer.mrmlScene.AddNode(thresholdedVolume)

        # Thresholding using CLI model
        slicer.cli.run(
            slicer.modules.thresholdscalarvolume,
            None,
            {'InputVolume': ct.GetID(),
            'OutputVolume': thresholdedVolume.GetID(),
            'ThresholdValue': 3000,
            'ThresholdType': 'Below'}, wait_for_completion=True,update_display=True)

        
        # Display the thresholded volume
        slicer.util.setSliceViewerLayers(background=thresholdedVolume)
        slicer.app.layoutManager().setLayout(3)  

        # Enable volume rendering
        slicer.modules.volumerendering.logic().CreateDefaultVolumeRenderingNodes(thresholdedVolume)

        # Get the slice view controller associated with the fourth view
        sliceController = slicer.app.layoutManager().sliceWidget('Red').sliceController()
        sliceController.fitSliceToBackground()


    def displayMask(self):
        maskVolume = self.inputMask.currentNode()
        if maskVolume is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Error", "Mask volume is not selected!")
            return
        slicer.util.setSliceViewerLayers(background=maskVolume)
        slicer.app.layoutManager().setLayout(3)  

        # Enable volume rendering
        slicer.modules.volumerendering.logic().CreateDefaultVolumeRenderingNodes(maskVolume)

        # Get the slice view controller associated with the fourth view
        sliceController = slicer.app.layoutManager().sliceWidget('Red').sliceController()
        sliceController.fitSliceToBackground()

    def detectElectrodes(self):
        volume = self.roivolume
        screw_list = self.roilist
        screw_points = self.getIJKCoordinatesFromFiducialList()
        screw_points = [arr[::-1] for arr in screw_points]
        self.electrode_names, electrode_lengths = self.extract_electrode_length()
        spacing = np.flip(self.inputCT.currentNode().GetSpacing())
        labels = self.logic_instance.labelVolume(volume, screw_list, screw_points, electrode_lengths, spacing)

        self.fiducialData = {}
        for electrode, point, length, name in zip(labels, screw_points, electrode_lengths, self.electrode_names):
            electrode = np.array(electrode)
            print(electrode.shape)
            newpoints, _, points_curve, offset = self.logic_instance.fitPointsOnElectrode(electrode, point, spacing, length)
            newpoints = [arr[::-1] for arr in newpoints]
            self.logic_instance.createFiducialListFromIJKCoordinates(newpoints,  self.inputCT.currentNode(), name)
            self.fiducialData[name] = {'curve': points_curve, 'offset': offset, 'number_of_points': length, 'spacing': spacing}#dict to use later


        
    def resliceImages(self):
        mri = self.inputMRI.currentNode()
        ct = self.inputCT.currentNode()
        # Create a new transform node
        transform_node = slicer.vtkMRMLTransformNode()
        slicer.mrmlScene.AddNode(transform_node)

        # Align the MRI volume to the CT volume
        mri.SetAndObserveTransformNodeID(transform_node.GetID())

        # Resample MRI volume to match CT volume's spacing and origin
        resliced_node = slicer.modules.volumes.logic().ResampleVolumeToReferenceVolume(mri, ct)
        return resliced_node

        

    def getIJKCoordinatesFromFiducialList(self):
        pointListNode = self.inputFiducial.currentNode()
        volumeNode = self.inputCT.currentNode()
        if pointListNode is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Error", "Points list is not selected!")
            return
        if volumeNode is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Error", "CT volume is not selected!")

        numPoints = pointListNode.GetNumberOfControlPoints()

        coordinates_list = []

        for markupsIndex in range(numPoints):
            # Get point coordinate in RAS
            point_Ras = [0, 0, 0]
            pointListNode.GetNthControlPointPositionWorld(markupsIndex, point_Ras)

            # If volume node is transformed, apply that transform to get volume's RAS coordinates
            transformRasToVolumeRas = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas)
            point_VolumeRas = transformRasToVolumeRas.TransformPoint(point_Ras)

            # Get voxel coordinates from physical coordinates
            volumeRasToIjk = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
            point_Ijk = [0, 0, 0, 1]
            volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas, 1.0), point_Ijk)
            point_Ijk = [int(round(c)) for c in point_Ijk[0:3]]
            coordinates_list.append(point_Ijk)

            # Print output
            print("Point", markupsIndex + 1, "voxel coordinates:", point_Ijk)
        return coordinates_list
    
    
    def extract_electrode_length(self):
        pointListNode = self.inputFiducial.currentNode()
        numPoints = pointListNode.GetNumberOfControlPoints()
        
        electrode_lengths = []
        electrode_names = []

        for markupsIndex in range(numPoints):
            # Get the name of the fiducial point
            fiducial_name = pointListNode.GetNthControlPointLabel(markupsIndex)
            
            # Extract the letter and number from the name
            match = re.match(r'([A-Za-z]+)-(\d+)', fiducial_name)
            if match:
                electrode_name = match.group(1)
                electrode_length = int(match.group(2))
            else:
                # If the format doesn't match, set default values
                electrode_name = ""
                electrode_length = 0

            # Append extracted letter and number to the respective lists
            electrode_names.append(electrode_name)
            electrode_lengths.append(electrode_length)
            
        return electrode_names, electrode_lengths

    def getROIFromFiducial(self):
        
        sourceVolumeNode = self.inputCT.currentNode()
        volumesLogic = slicer.modules.volumes.logic()
        volumeNode = volumesLogic.CloneVolume(slicer.mrmlScene, sourceVolumeNode, "Cloned volume")
        
        # Get dimensions of the volume
        radius = 10
        dimensions = volumeNode.GetImageData().GetDimensions()
        spacing = volumeNode.GetSpacing()
        print(spacing)
        # Get the volume as a numpy array
        volume_array = slicer.util.arrayFromVolume(volumeNode)

        # Create a mask volume with zeros
        mask_array = np.zeros(dimensions[::-1], dtype=np.uint8)  # Reversed dimensions
        ROI_list = []

        # Iterate over the list of coordinates
        coordinateList = self.getIJKCoordinatesFromFiducialList()
        i = 0
        for coordinate in coordinateList:
            i += 1
            # Create grid of coordinates for each dimension
            x_indices, y_indices, z_indices = np.meshgrid(np.arange(dimensions[0]), np.arange(dimensions[1]), np.arange(dimensions[2]), indexing='ij')

            # Calculate distance from the center of the sphere
            distance = np.sqrt(((x_indices - coordinate[0]) * spacing[0])**2 + ((y_indices - coordinate[1]) * spacing[1])**2 + ((z_indices - coordinate[2]) * spacing[2])**2)
            # Transpose distance array to match the shape of mask_array
            distance = np.transpose(distance, axes=(2, 1, 0))

            # Update mask_array based on distance and radius
            roi = (distance <= radius).astype(np.uint8)
            roi = self.logic_instance.bw(roi*volume_array)
            # ball_array = volume_array*roi
            # file_name = f"array_policko_{i}.npy"
            # np.save(file_name, ball_array)

            mask_array += roi

            ROI_list.append(roi)

        mask_array = np.clip(mask_array, 0, 1)
        # Apply the mask
        volume_array *= mask_array
        
        # Update the volume node with the modified array
        array = slicer.util.arrayFromVolume(self.inputMask.currentNode()) #tady bude ze vezmu to masked
        array = self.logic_instance.maskVolume(slicer.util.arrayFromVolume(sourceVolumeNode), array)

        volume_array += array

        slicer.util.updateVolumeFromArray(volumeNode, volume_array)
        slicer.cli.run(
            slicer.modules.thresholdscalarvolume,
            None,
            {'InputVolume': volumeNode,
            'OutputVolume': volumeNode,
            'ThresholdValue': 3000,
            'ThresholdType': 'Below'}, wait_for_completion=True,update_display=False)
        
        self.roivolume = slicer.util.arrayFromVolume(volumeNode)
        self.roilist = ROI_list
        print("suc")

    def modifyMask(self):
        # Get arrays from the volumes
        scalarVolumeArray = slicer.util.arrayFromVolume(scalarVolumeNode)
        labelMapArray = slicer.util.arrayFromVolumeAsImage(labelMapNode)

        # Apply the mask
        maskedVolumeArray = scalarVolumeArray * (labelMapArray > 0)

        # Create a new volume node to store the masked volume
        maskedVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        slicer.util.updateVolumeFromArray(maskedVolumeNode, maskedVolumeArray)
    



        


                


#
# beta1Logic
#

class beta1Logic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return beta1ParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def determineVariance(self, x, y, z):
        variances = [np.var(axis) for axis in (x, y, z)]
        max_variance_index = np.argmax(variances)
        return max_variance_index


    def generatePolynomial(self, x, y, z, intensity, max_variance_index, poly_order):
        intesity = intensity[x,y,z]
        if max_variance_index == 0:
            poly_coefficients_xy = np.polyfit(x, y, poly_order, w=intesity)
            poly_coefficients_xz = np.polyfit(x, z, poly_order, w=intesity)
            return poly_coefficients_xy, poly_coefficients_xz, 'X'
        elif max_variance_index == 1:
            poly_coefficients_xy = np.polyfit(y, x, poly_order, w=intesity)
            poly_coefficients_xz = np.polyfit(y, z, poly_order, w=intesity)
            return poly_coefficients_xy, poly_coefficients_xz, 'Y'
        else:
            poly_coefficients_xy = np.polyfit(z, x, poly_order, w=intesity)
            poly_coefficients_xz = np.polyfit(z, y, poly_order, w=intesity) 
            return poly_coefficients_xy, poly_coefficients_xz, 'Z'

    def generateCentralAxis(self, volume, poly_order):
        x, y, z = np.nonzero(volume)
        max_variance_index = self.determineVariance(x, y, z)
        poly_coefficients_xy, poly_coefficients_xz, _ = self.generatePolynomial(x, y, z, volume, max_variance_index, poly_order)
        
        
        if max_variance_index == 0:
            x_start = min(x)
            x_end = max(x)
            central_axis_start = (x_start, np.polyval(poly_coefficients_xy, x_start), np.polyval(poly_coefficients_xz, x_start))
            central_axis_end = (x_end, np.polyval(poly_coefficients_xy, x_end), np.polyval(poly_coefficients_xz, x_end))
        elif max_variance_index == 1:
            y_start = min(y)
            y_end = max(y)
            central_axis_start = (np.polyval(poly_coefficients_xy, y_start), y_start, np.polyval(poly_coefficients_xz, y_start))
            central_axis_end = (np.polyval(poly_coefficients_xy, y_end), y_end, np.polyval(poly_coefficients_xz, y_end))
        else:
            z_start = min(z)
            z_end = max(z)
            central_axis_start = (np.polyval(poly_coefficients_xy, z_start), np.polyval(poly_coefficients_xz, z_start), z_start)
            central_axis_end = (np.polyval(poly_coefficients_xy, z_end), np.polyval(poly_coefficients_xz, z_end), z_end)

        # Calculate the center point of the matrix
        center = np.array(volume.shape) / 2
        vector_to_center = center - np.array(central_axis_start)
        central_axis_vector = np.array(central_axis_end) - np.array(central_axis_start)
        
        # Check if the vector points towards the center, if not, flip it
        if np.dot(vector_to_center, central_axis_vector) < 0:
            central_axis_start, central_axis_end = central_axis_end, central_axis_start

        return central_axis_start, central_axis_end
    
    def labelVolume(self, volume, screws, screw_points, electrodes_parameter_list, spacing):
        electrodes_vectors_starts = []
        electrodes_vectors_ends = []
        i = -1
        for electrode_parametr,screw in zip(electrodes_parameter_list, screws):
            i += 1
            screw = volume*screw
            screw_vector_start, screw_vector_end = self.generateCentralAxis(screw, 1)
            
            screw_vector = (np.array(screw_vector_end) - np.array(screw_vector_start))/np.linalg.norm(np.array(screw_vector_end) - np.array(screw_vector_start))
            
            lenght = 1.5*(2*electrode_parametr + 1.5*(electrode_parametr-1))
            
            electrode_vector_start = np.array(screw_vector_end)
            electrodes_vector_end = (np.array(electrode_vector_start + (screw_vector * lenght), dtype = int))
            electrodes_vectors_starts.append(electrode_vector_start)
            electrodes_vectors_ends.append(electrodes_vector_end)


        volume_2d = np.transpose(np.nonzero(volume))

        covariances = []
        midpoints = []
        # Calculate covariances and midpoints for each vector
        for start, end in zip(electrodes_vectors_starts, electrodes_vectors_ends):
            
            midpoint = (start + end) / 2
            midpoints.append(midpoint)
            
            data = np.vstack((start, end))
            covariance = np.cov(data, rowvar=False)
            epsilon = 0.01 * np.max(np.diag(covariance))
            covariances.append(np.linalg.inv(covariance + epsilon * np.eye(3)))
        
        # Initialize GMM 
        print('initializing GMM')
        n_components = len(electrodes_vectors_starts)

        gmm = GaussianMixture(n_components=n_components, covariance_type='full', 
                            means_init=np.array(midpoints), 
                            precisions_init=np.array(covariances), 
                            weights_init=np.full(n_components, 1/n_components))

        # Fit the model to the data
        gmm.fit(volume_2d)
        # Predict cluster labels for the data
        labels = gmm.predict(volume_2d)
        print('gmm completed')

        unique_labels = np.unique(labels)
        separated_electrodes = []

        for label in unique_labels:
            # Filter out the rows corresponding to the current label
            label_indices = np.where(labels == label)[0]
            label_voxels = volume_2d[label_indices]

            # Create a boolean mask for voxels labeled as the current label
            mask = np.zeros_like(volume, dtype=bool)
            mask[tuple(label_voxels.T)] = True
            
            selected_voxels = np.zeros_like(volume)
            selected_voxels[mask] = volume[mask]
            separated_electrodes.append(selected_voxels)

        return separated_electrodes
    

    def fitPointsOnElectrode(self, electrode, point, spacing, electrode_lenght):    
        print('starting fitting points on electrode')
        
        x, y, z = np.nonzero(electrode)
        max_variance = self.determineVariance(x,y,z)
        poly_coeficients_1, poly_coeficients_2, variance = self.generatePolynomial(x,y,z, electrode, max_variance, 5)

        furthest_point = np.array(point)
        nonzero_indices = np.transpose(np.nonzero(electrode))
        closest_index = nonzero_indices[np.argmax(cdist([furthest_point], nonzero_indices))]
        
        if variance == 'X':
            x_min = closest_index[0]
            x_max = furthest_point[0]
            x_values = np.linspace(x_min, x_max, 1000)

            y_values = np.polyval(poly_coeficients_1, x_values)
            z_values = np.polyval(poly_coeficients_2, x_values)

        elif variance == 'Y':
            y_min = closest_index[1]
            y_max = furthest_point[1]
            y_values = np.linspace(y_min, y_max, 1000)

            x_values = np.polyval(poly_coeficients_1, y_values)
            z_values = np.polyval(poly_coeficients_2, y_values)

        elif variance == 'Z':
            z_min = closest_index[2]
            z_max = furthest_point[2]
            z_values = np.linspace(z_min, z_max, 1000)

            x_values = np.polyval(poly_coeficients_1, z_values)
            y_values = np.polyval(poly_coeficients_2, z_values)

        curve_points = np.column_stack((x_values, y_values, z_values))

        offsets = []
        correlation_values = []

        cropped_array = self.cropArray(electrode)
        for i in range (50):
            offset = 2*i
            selected_points = self.selectPointsFromCurve(curve_points, electrode_lenght, spacing, offset)
            gaussian_array = self.createArrayWithGaussians(electrode, selected_points, 1)
            
            cropped_gaussian_array = self.cropArrayAsArray(electrode, gaussian_array)

            correlation = np.mean(np.corrcoef(cropped_array.flatten(),cropped_gaussian_array.flatten())[0, 1])
            
            offsets.append(offset)
            correlation_values.append(correlation)

        correlation_values = uniform_filter(correlation_values, 20)
        max_index = np.argmax(correlation_values)
        correlation_values = self.penalty_function(correlation_values, 0.0035)
        max_index_after_penalty = np.argmax(correlation_values)

        selected_points =  self.selectPointsFromCurve(curve_points, electrode_lenght, spacing, max_index ) 

        is_max_first_peak = True
        if max_index != max_index_after_penalty:
            is_max_first_peak = False

        return selected_points, is_max_first_peak, curve_points, max_index_after_penalty

    def cropArray(self, array):
        non_zero_indices = np.nonzero(array)
        min_indices = np.min(non_zero_indices, axis=1)
        max_indices = np.max(non_zero_indices, axis=1)

        small_array = array[min_indices[0]:max_indices[0]+1, 
                                min_indices[1]:max_indices[1]+1, 
                                min_indices[2]:max_indices[2]+1]
        return small_array
    
    def cropArrayAsArray(self, array, array2):
        if array.shape != array2.shape:
            print('arrays shapes have to match!')
            return -1
        non_zero_indices = np.nonzero(array)
        min_indices = np.min(non_zero_indices, axis=1)
        max_indices = np.max(non_zero_indices, axis=1)

        small_array = array2[min_indices[0]:max_indices[0]+1, 
                                min_indices[1]:max_indices[1]+1, 
                                min_indices[2]:max_indices[2]+1]
        return small_array

    def selectPointsFromCurve(self, curve_points, number_of_points, spacing, offset = 0):
        starting_index = offset
        desired_distance = 3.5 #3,5 millimetres
        point_distances = np.linalg.norm(np.diff(curve_points, axis=0) * spacing, axis=1)

        selected_points = [curve_points[starting_index]]
        total_distance = 0
        points_selected = 1
        
        for i, distance in enumerate(point_distances[starting_index:], start=starting_index):
            if points_selected == number_of_points: break
            total_distance += distance
            if total_distance >= desired_distance:
                selected_points.append(curve_points[i])
                total_distance = 0
                points_selected += 1

        selected_points = np.array(selected_points)
        return selected_points
    
    def createGaussianBall(self, array, center, sigma):
        x, y, z = center
        x_indices = np.arange(max(0, x - 3 * sigma), min(array.shape[0], x + 3 * sigma))
        y_indices = np.arange(max(0, y - 3 * sigma), min(array.shape[1], y + 3 * sigma))
        z_indices = np.arange(max(0, z - 3 * sigma), min(array.shape[2], z + 3 * sigma))

        x_slice = slice(max(0, x - 3 * sigma), min(array.shape[0], x + 3 * sigma))
        y_slice = slice(max(0, y - 3 * sigma), min(array.shape[1], y + 3 * sigma))
        z_slice = slice(max(0, z - 3 * sigma), min(array.shape[2], z + 3 * sigma))

        xx, yy, zz = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        distances = ((xx - x)**2 + (yy - y)**2 + (zz - z)**2)**0.5
        gaussian_ball = np.exp(-(distances**2) / (2 * sigma**2))
        array[x_slice, y_slice, z_slice] += gaussian_ball

    def createArrayWithGaussians(self, array, points_list, ball_radius):
        gaussian_array = np.zeros(array.shape, dtype=float)
        for i in range(3):
                point = points_list[i]
                rounded_point = tuple(np.round(point).astype(int))
                self.createGaussianBall(gaussian_array, rounded_point, ball_radius)
        return gaussian_array

    def penalty_function(self, values, reduction_percentage):

        penalized_values = []
        for i, value in enumerate(values):
            penalty = 1 - (i * reduction_percentage)
            penalized_values.append(value * penalty)
        return penalized_values
    
    def bw(self, array):
        binary_array = (array > 1300).astype(int)
        labeled_data = measure.label(binary_array) 
        region_sizes = np.bincount(labeled_data.ravel())
        largest_label = np.argmax(region_sizes[1:]) + 1
        largest_object = np.array(labeled_data == largest_label, dtype=np.uint8)

        return largest_object
    
    def shiftPoints(self, listNode, shift, volumeNode, dict):
        
        if listNode is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Error", "Please select an electrode.")
            return
        name = listNode.GetName()
        # Get curve and offset associated with selected electrode
        data = dict[name]
        curve_points = data['curve']
        offset = data['offset']
        number_of_points = data['number_of_points']
        spacing = data['spacing']
        
        # Get user-defined shift amount
        shift_amount = np.array([shift], dtype=np.int64)
        
        # Shift points
        shifted_points = self.selectPointsFromCurve(curve_points, number_of_points, spacing, offset + shift_amount[0])
        shifted_points = [arr[::-1] for arr in shifted_points]
        # Remove old fiducial node from scene
        slicer.mrmlScene.RemoveNode(listNode)
        
        # Create new fiducial list from shifted points
        newListNode = self.createFiducialListFromIJKCoordinates(shifted_points, volumeNode, name)
        
        # Update fiducialNodes dictionary with the new fiducial node
        data['offset'] = offset + shift_amount[0]
        dict[name] = data
        return dict, newListNode
        

    def createFiducialListFromIJKCoordinates(self, coordinates_list, volumeNode, name):
        pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        pointListNode.SetName(name)

        # Get the inverse transform from IJK to RAS
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRasMatrix)

        i = 0
        for point_Ijk in coordinates_list:
            point_Ijk = point_Ijk.tolist()
            point_Ijk.append(1)
            # Convert IJK to RAS
            point_Ras = ijkToRasMatrix.MultiplyFloatPoint(point_Ijk)
            # Add the point to the fiducial list
            fiducial_name = f"{name}{i+1}"
            pointListNode.AddControlPoint(point_Ras[0:3], fiducial_name)
            i += 1
        
        return pointListNode
    
    def combineFiducialLists(self, dict, volumeNode, names):
        pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        pointListNode.SetName('pointsFinal')

        # Get the inverse transform from IJK to RAS
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRasMatrix)

        for name in names:
            data = dict[name]
            curve_points = data['curve']
            offset = data['offset']
            number_of_points = data['number_of_points']
            spacing = data['spacing']
            coordinates_list = self.selectPointsFromCurve(curve_points, number_of_points, spacing, offset)
            coordinates_list = [arr[::-1] for arr in coordinates_list]

            i = 0
            for point_Ijk in coordinates_list:
                point_Ijk = point_Ijk.tolist()
                point_Ijk.append(1)
                # Convert IJK to RAS
                point_Ras = ijkToRasMatrix.MultiplyFloatPoint(point_Ijk)
                # Add the point to the fiducial list
                fiducial_name = f"{names[i]}{i+1}"
                pointListNode.AddControlPoint(point_Ras[0:3], fiducial_name)
                i += 1
            slicer.mrmlScene.RemoveNode(slicer.util.getNode(name))
        
        
        return pointListNode

    def maskVolume(self, volume, mask_volume):
        if volume.shape != mask_volume.shape:
            raise ValueError("Volume and mask must have the same shape.")

        masked_volume = np.where(mask_volume != 0, volume, 0)

        return masked_volume
        
    #
# beta1Test
#

class beta1Test(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_beta11()

    def test_beta11(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('beta11')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = beta1Logic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')

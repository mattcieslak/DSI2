#!/usr/bin/env python

import numpy as np
import nibabel as nib
from scipy.io.matlab import savemat
from dsi2.streamlines.track_dataset import TrackDataset
from dsi2.streamlines.track_math import voxels_to_streamlines,region_pair_dict_from_roi_list,trackvis_header_from_info
from dsi2.volumes import QSDR_AFFINE, QSDR_SHAPE, QSDR_VOXEL_SIZE
import os
import random
import os.path as op
from dsi2.volumes import QSDR_SHAPE, QSDR_AFFINE, find_graphml_from_filename, load_lausanne_graphml
from dsi2.database.traited_query import Scan, TrackLabelSource, TrackScalarSource
from dsi2.aggregation import make_aggregator
import pytest
from dsi2.aggregation import make_aggregator
from paths import test_input_data
from dsi2.ui.local_data_importer import LocalDataImporter
from dsi2.database.track_datasource import TrackDataSource


n_streamlines = 5
def simulated_atlas(scale=33,volume_shape=QSDR_SHAPE, 
                    volume_affine=QSDR_AFFINE ):
    graphml = find_graphml_from_filename("scale%d" % scale)
    node_data = load_lausanne_graphml(graphml)
    regions = node_data['regions']
    
    subvolume = np.array(volume_shape) - 2
    available_voxels = np.prod(subvolume)
    voxels_per_region = available_voxels // len(regions)
    if voxels_per_region == 0: 
        raise ValueError("Volume is not big enough to contain all regions")
    
    label_data = np.zeros(available_voxels)
    labels = np.repeat(regions, voxels_per_region)
    label_data[:len(labels)] = labels
    
    new_atlas = np.zeros(volume_shape)
    new_atlas[1:-1,1:-1,1:-1] = label_data.reshape(subvolume)
    return nib.Nifti1Image(new_atlas,volume_affine), regions
    
def simulated_connection(nib_atlas_img, output_trk_filename="",
                        STREAMLINE_ORI="LPS",
                        n_streamlines=5, curvy=False, points_per_streamline=10,
                        return_streamlines=False):
    """
    pick two regions to create a streamlines between
    """
    label_cube = nib_atlas_img.get_data()
    candidate_regions = np.unique(nib_atlas_img.get_data()[nib_atlas_img.get_data() > 0])
    target_regions = np.random.choice(candidate_regions, size=2, replace=False)
    test_conn_id = region_pair_dict_from_roi_list(candidate_regions)[tuple(sorted(target_regions))]
    regA_voxels = [np.array(np.nonzero(label_cube == target_regions[0])).T]
    regA_streamline_coords = voxels_to_streamlines(regA_voxels,nib_atlas_img,STREAMLINE_ORI)[0]
    regB_voxels = [np.array(np.nonzero(label_cube == target_regions[1])).T]
    regB_streamline_coords = voxels_to_streamlines(regB_voxels,nib_atlas_img,STREAMLINE_ORI)[0]
    
    streamlines = []
    regA_choices = np.random.choice(np.arange(len(regA_streamline_coords)),n_streamlines)
    regB_choices = np.random.choice(np.arange(len(regB_streamline_coords)),n_streamlines)
    for coord_indexA, coord_indexB in  zip(regA_choices,regB_choices):
        coordA = regA_streamline_coords[coord_indexA]
        coordB = regB_streamline_coords[coord_indexB]
        if curvy:
            raise NotImplementedError("someday...")
        else:
            npoints = np.random.randint(points_per_streamline -2, points_per_streamline+2)
            streamlines.append(
                np.array([
                               np.linspace(coordA[0],coordB[0],npoints),
                               np.linspace(coordA[1],coordB[1],npoints),
                               np.linspace(coordA[2],coordB[2],npoints)
                               ]).T
                )                
    header = trackvis_header_from_info( STREAMLINE_ORI, 
                                        nib_atlas_img.shape, nib_atlas_img.get_header().get_zooms())
                                        
    nib.trackvis.write(output_trk_filename, [(sl,None,None) for sl in streamlines], hdr_mapping=header)    
    if return_streamlines:
        return target_regions, streamlines
    return target_regions

def create_fake_fib_file(fname, 
                                             volume_shape = np.array([50,50,50]),
                                             voxel_size=np.array([2.0,2.0,2.0]),
                                             n_odfs=10,
                                             odf_resolution=3
                                             ):
    """
    Creates a small fib.gz file
    """
    import gzip
    from scipy.io.matlab import savemat
    from dipy.core.subdivide_octahedron import create_unit_sphere
    
    sphere = create_unit_sphere(odf_resolution)
    fa0 = np.zeros(np.prod(volume_shape))
    index0 = np.zeros(np.prod(volume_shape))
    fa0[:n_odfs] = 1
    index0[:n_odfs] = 1
    
    fop = gzip.open(fname,"wb")
    savemat(fop, {
            "dimension": volume_shape,
            "fa0": fa0,
            "odf_vertices": sphere.vertices,
            "odf_faces": sphere.faces,
            "index0":index0,
            "voxel_size":voxel_size
            },
        format="4"
        )
    fop.close()
    

def simulated_subject(droot,subject_number):
    subjname = "S%03d" % subject_number
    subjdir = op.join(droot,subjname)
    trk_pth = op.join(subjdir,subjname+".trk")
    fib_pth = op.join(subjdir,subjname+".fib.gz")
    pkl_trk_pth = op.join(subjdir,subjname+".pkl.trk")
    if op.exists(subjdir): os.system("rm -rf " + subjdir)
    os.makedirs(subjdir)
    pkl_path=op.join(subjdir,subjname+".pkl")
    
    return  Scan(
            scan_id=subjname,
            subject_id=subjname,
            study="testing",
            scan_group="fake_brain",
            scan_gender=random.choice(["F","M"]),
            #streamline_space="qsdr",
            software="DSI Studio",
            reconstruction="gqi",
            #fib_file=fib,
            trk_file=trk_pth,
            pkl_path=pkl_path,
            pkl_trk_path=pkl_trk_pth
    )
    
def generate_scalars_files(streamlines,out_txtfile_name):
    """
    generates text files like DSI Studio for GFA/QA/etc
    """
    fop = open(out_txtfile_name,"w")
    for nstream, stream in enumerate(streamlines):
        fop.write(" ".join([str(nstream)] * len(stream)) + "\n")
    fop.close()

def simulated_custom_template_subject(droot,subject_number,template_volume_path,
                                      volume_affine,volume_shape,
                                      volume_voxel_size):
    # Set the streamline space and path to custom template
    scan = simulated_subject(droot, subject_number)
    subjdir = op.join(droot,scan.scan_id)
    scan.streamline_space = "custom template"
    scan.template_volume_path = template_volume_path
    scan.template_affine = volume_affine
    scan.template_volume_shape = volume_shape
    scan.template_voxel_size = volume_voxel_size
    
    # generate the atlases
    for scale in [33,60,125,250]:
        template_nii_path = op.join(subjdir,scan.scan_id+".scale%d.nii.gz" % scale)
        img,_ = simulated_atlas(scale=scale, volume_shape=volume_shape, 
                        volume_affine=volume_affine)
        img.to_filename(template_nii_path)
        scan.track_label_items.append(
                    TrackLabelSource(
                        name="Lausanne2008",
                        parameters={"scale":scale,"thick":1},
                        numpy_path=op.join(subjdir,"scale%d.npy" % scale),
                        template_volume_path=template_nii_path
                        )
                    )
        
    # generate the streamlines
    chosen_atlas = random.choice(scan.track_label_items)
    targeted_regions, streamlines = simulated_connection(
                  nib.load(chosen_atlas.template_volume_path),
                  output_trk_filename=scan.trk_file, return_streamlines=True)
    # Attach the correct answer to the chosen label dataset
    chosen_atlas.targeted_regions = targeted_regions
    
    # Generate some scalar data
    for scalar in ["gfa", "qa"]:
        scalar_txt = op.join(subjdir, scalar+".txt")
        generate_scalars_files(streamlines,scalar_txt)
        scan.track_scalar_items.append(
            TrackScalarSource(
                name=scalar,
                txt_path=scalar_txt,
                numpy_path=scalar_txt + ".npy"
                )
            )
    return scan

def check_scalars(scan):
    for scalar in scan.track_scalar_items:
        assert op.exists(scalar.numpy_path)
        vals = scalar.get_scalars()
        assert np.all(vals == np.arange(len(vals)))
    return True


def check_mapping(scans, test_data_importer):
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for \
                                            scan in test_data_importer.datasets])
    for nscan,scan in enumerate(scans):
        for label in scan.track_label_items:
            if not hasattr(label,"targeted_regions"): continue
            agg = make_aggregator(
                            algorithm="region labels",
                            atlas_name="Lausanne2008",
                            atlas_scale=label.parameters['scale'],
                            data_source=tds)
            region_pair = tuple(map(int,sorted(label.targeted_regions)))
            region_pair_id = agg.region_pairs_to_index[region_pair]
            agg.query_track_source_with_region_pair(region_pair_id)
            agg.update_clusters()
            assert agg.track_sets[nscan].get_ntracks() == n_streamlines
            
            
volume_shape=np.array([50,50,50])            
volume_affine = np.array([[2.,0.,0.,0.],
                                           [0.,2.,0.,0.],
                                           [0.,0.,2.,0.],
                                           [0.,0.,0.,1.]])
volume_voxel_size= np.array([2.,2.,2.])
nscans=5

def test_custom_template():
    
    json_out = os.path.join(test_input_data,"custom_template.json")
    
    custom_template_vol = nib.Nifti1Image(np.zeros(volume_shape),volume_affine)
    custom_template_filename = op.join(test_input_data, "custom_template.nii.gz")
    custom_template_vol.to_filename(custom_template_filename)
    
    scans = [simulated_custom_template_subject(test_input_data,x,custom_template_filename, volume_affine,
                                volume_shape,volume_voxel_size) for x in xrange(nscans)]    
    
    # Test the local data importer
    ldi = LocalDataImporter(datasets=scans)
    ldi.process_inputs()
    ldi.json_file = json_out
    ldi.save_json()
    
    for scan in ldi.datasets:
        assert check_scalars(scan)
    
    test_data_importer = LocalDataImporter(json_file=json_out)
    
    check_mapping(scans, test_data_importer)
    
def simulated_qsdr_subject(droot, **kwargs):
    scan = simulated_subject(droot,**kwargs)
    scan.fib_file= op.join(subjdir,subjname+".fib.gz")
    create_fake_fib_file(scan.fib_file, volume_shape=QSDR_SHAPE,
                                        voxel_size=QSDR_VOXEL_SIZE)
    
    for scale in [33,60,125,250]:
        b0_nii = op.join(subjdir,subjname+".scale%d.nii.gz" % scale)
        qsdr_nii = op.join(subjdir,subjname+".scale%d.QSDR.nii.gz" % scale)
        get_scale60_nim().to_filename(scale60)
        sc.track_label_items.append(
                    TrackLabelSource(
                        name="Lausanne2008",
                        parameters={"scale":33,"thick":1},
                        numpy_path=op.join(subjdir,"scale33.npy"),
                        b0_volume_path=scale33,
                        qsdr_volume_path=scale33_q
                        )
                    )
    return scan
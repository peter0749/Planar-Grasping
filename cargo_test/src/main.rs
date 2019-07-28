#![feature(core_intrinsics)]
#![feature(proc_macro_hygiene)]
use inline_python::{pyo3, python};
use numpy::{PyArray, PyArray1, PyArray2, PyArrayDyn, get_array_module};

fn print_type_of<T>(_: &T) {
    println!("{}", unsafe { std::intrinsics::type_name::<T>()  });
}

fn main() {
    let img_path = "/media/peter/085C6EBC5C6EA462/CGD/example_imgs/rgb1.jpg";
    let depth_path = "/media/peter/085C6EBC5C6EA462/CGD/example_imgs/depth1.jpg";
    let python_context = inline_python::Context::new();
    python! {
        #![context = &python_context]
        import numpy as np
        import cv2
        from grasp_baseline.inference import GraspDetector
        detector = GraspDetector()
    }
    python! {
        #![context = &python_context]
        img = cv2.imread('img_path, cv2.IMREAD_COLOR)[...,::-1]
        depth = cv2.imread('depth_path, cv2.IMREAD_GRAYSCALE)
        grasps, degrees, graspscores, object_bounding_boxes, categories, yolo_scores = detector.detect([img], [depth])
        single_grasp = None
        if len(grasps)>0 and len(grasps[0])>0 and len(grasps[0][0])>0:
            single_grasp = grasps[0][0][0].astype(np.float32)
    }
    let gil = pyo3::Python::acquire_gil();
    let py  = gil.python();
    let py_global = python_context.globals(py);
    let grasp_pts = py_global.get_item("grasps").unwrap(); // Still figuring out how to use rust-numpy...
    println!("Proposed grasps:\n{:?}", Some(grasp_pts));
    let single_grasp = py_global.get_item("single_grasp").unwrap(); // Still figuring out how to use rust-numpy...
    println!("Most possible grasp:\n{:?}", Some(single_grasp));
}

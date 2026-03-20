use ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

pub fn numpy_to_array2(arr: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
	arr.as_array().to_owned()
}

pub fn numpy_to_array1(arr: PyReadonlyArray1<'_, f64>) -> Array1<f64> {
	arr.as_array().to_owned()
}

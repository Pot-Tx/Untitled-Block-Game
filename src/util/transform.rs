use glam::*;
use num_traits::Num;

pub trait Trans {
    type Scalar: Num;
}

pub trait Trans3: Trans {
    fn rotation(yaw: Self::Scalar, pitch: Self::Scalar, roll: Self::Scalar) -> Self;
}

pub trait Trans4: Trans {
    fn translation(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self;

    fn rotation(yaw: Self::Scalar, pitch: Self::Scalar, roll: Self::Scalar) -> Self;

    fn projection(
        near: Self::Scalar,
        far: Self::Scalar,
        fov: Self::Scalar,
        aspect: Self::Scalar,
    ) -> Self;
}

macro_rules! impl_trans_for {
    ($($mat:ty : $scalar:ty),* $(,)?) => {
        $(
            impl Trans for $mat {
                type Scalar = $scalar;
            }
        )*
    };
}

impl_trans_for!(Mat2: f32, Mat3: f32, Mat3A: f32, Mat4: f32);
impl_trans_for!(DMat2: f64, DMat3: f64, DMat4: f64);

macro_rules! impl_trans3_for {
    ($($mat:ty),* $(,)?) => {
	    $(
	        impl Trans3 for $mat {
		        fn rotation(yaw: Self::Scalar, pitch: Self::Scalar, roll: Self::Scalar) -> Self {
			        let (cy, sy, cp, sp, cr, sr) = (
                        yaw.cos(),
                        yaw.sin(),
                        pitch.cos(),
                        pitch.sin(),
                        roll.cos(),
                        roll.sin(),
                    );
			        Self::from_cols_array(&[
		                cy * cr - sy * cp * sr, -sp * sr, -sy * cr - cy * cp * sr,
		                -sy * sp, cp, -cy * sp,
		                sy * cp * cr + cy * sr, sp * cr, cy * cp * cr - sy * sr,
	                ])
		        }
	        }
	    )*
    };
}

impl_trans3_for!(Mat3, Mat3A, DMat3);

macro_rules! impl_trans4_for {
    ($($mat:ty),* $(,)?) => {
        $(
            impl Trans4 for $mat {
	            fn translation(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self {
	                Self::from_cols_array(&[
		                1.0, 0.0, 0.0, 0.0,
		                0.0, 1.0, 0.0, 0.0,
		                0.0, 0.0, 1.0, 0.0,
		                x, y, z, 1.0,
	                ])
                }

	            fn rotation(yaw: Self::Scalar, pitch: Self::Scalar, roll: Self::Scalar) -> Self {
	                let (cy, sy, cp, sp, cr, sr) = (
                        yaw.cos(),
                        yaw.sin(),
                        pitch.cos(),
                        pitch.sin(),
                        roll.cos(),
                        roll.sin(),
                    );
	                Self::from_cols_array(&[
		                cy * cr - sy * cp * sr, -sy * sp, sy * cp * cr + cy * sr, 0.0,
		                -sp * sr, cp, sp * cr, 0.0,
		                -sy * cr - cy * cp * sr, -cy * sp, cy * cp * cr - sy * sr, 0.0,
		                0.0, 0.0, 0.0, 1.0,
	                ])
                }

	            fn projection(near: Self::Scalar, far: Self::Scalar, fov: Self::Scalar, aspect: Self::Scalar) -> Self {
                    let (tanf, range) = ((fov / 2.0).tan(), far - near);
                    Self::from_cols_array(&[
	                    1.0 / (aspect * tanf), 0.0, 0.0, 0.0,
	                    0.0, 1.0 / tanf, 0.0, 0.0,
	                    0.0, 0.0, near / range, -1.0,
	                    0.0, 0.0, near * far / range, 0.0,
                    ])
                }
            }
        )*
    };
}

impl_trans4_for!(Mat4, DMat4);

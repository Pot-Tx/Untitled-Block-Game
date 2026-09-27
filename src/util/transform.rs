use crate::render::ViewPortAlignment;
use glam::*;
use num_traits::Num;

/// A matrix with a scalar type, the common part of the two transform traits.
pub trait Trans {
    type Scalar: Num;
}

/// A 3x3 matrix that can be built from an intrinsic rotation.
/// A 3x3 matrix that can be built from a rotation around the three axes.
pub trait Trans3: Trans {
    /// Builds the rotation of `yaw` around the y axis, `pitch` around the x axis
    /// and `roll` around the z axis; the angles are in radians.
    fn rotation(yaw: Self::Scalar, pitch: Self::Scalar, roll: Self::Scalar) -> Self;
}

/// A 4x4 matrix that can be built from the usual transformations.
pub trait Trans4: Trans {
    /// Builds the matrix that translates by `(x, y, z)`.
    fn translation(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self;

    /// Builds the rotation of `yaw` around the y axis, `pitch` around the x axis
    /// and `roll` around the z axis; the angles are in radians.
    fn rotation(yaw: Self::Scalar, pitch: Self::Scalar, roll: Self::Scalar) -> Self;

    /// Builds the perspective projection of a camera with the given `near` and
    /// `far` planes, vertical field of view `fov` and `aspect` ratio.
    ///
    /// Depth is reversed, so the near plane maps to 1 and the far plane to 0,
    /// which matches the `Greater` depth comparison of the render batches.
    fn projection(
        near: Self::Scalar,
        far: Self::Scalar,
        fov: Self::Scalar,
        aspect: Self::Scalar,
    ) -> Self;

    /// Builds the matrix that maps a `width` by `height` viewport onto the
    /// surface, anchored to `alignment`.
    fn viewport(
        width: Self::Scalar,
        height: Self::Scalar,
        alignment: ViewPortAlignment,
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

	            fn viewport(width: Self::Scalar, height: Self::Scalar, alignment: ViewPortAlignment, aspect: Self::Scalar)
	            -> Self {
                    // Viewports are square, so the window aspect shrinks the
                    // projection along whichever axis is longer; each alignment
                    // then anchors that square to a different window edge.
                    let fit_x = if aspect > 1.0 { 1.0 / aspect } else { 1.0 };
                    let fit_y = if aspect < 1.0 { aspect } else { 1.0 };

		            match alignment {
			            ViewPortAlignment::Middle => Self::from_cols_array(&[
				            2.0 * fit_x / width, 0.0, 0.0, 0.0,
				            0.0, -2.0 * fit_y / height, 0.0, 0.0,
				            0.0, 0.0, 1.0, 0.0,
				            -fit_x, fit_y, 0.0, 1.0,
			            ]),

			            ViewPortAlignment::Left => Self::from_cols_array(&[
				            2.0 * fit_x / width, 0.0, 0.0, 0.0,
				            0.0, -2.0 * fit_y / height, 0.0, 0.0,
				            0.0, 0.0, 1.0, 0.0,
				            -1.0, fit_y, 0.0, 1.0,
			            ]),

			            ViewPortAlignment::Right => Self::from_cols_array(&[
				            2.0 * fit_x / width, 0.0, 0.0, 0.0,
				            0.0, -2.0 * fit_y / height, 0.0, 0.0,
				            0.0, 0.0, 1.0, 0.0,
				            1.0 - 2.0 * fit_x, fit_y, 0.0, 1.0,
			            ]),

			            ViewPortAlignment::Bottom => Self::from_cols_array(&[
				            2.0 * fit_x / width, 0.0, 0.0, 0.0,
				            0.0, -2.0 * fit_y / height, 0.0, 0.0,
				            0.0, 0.0, 1.0, 0.0,
				            -fit_x, 2.0 * fit_y - 1.0, 0.0, 1.0,
			            ]),

			            ViewPortAlignment::Up => Self::from_cols_array(&[
				            2.0 * fit_x / width, 0.0, 0.0, 0.0,
				            0.0, -2.0 * fit_y / height, 0.0, 0.0,
				            0.0, 0.0, 1.0, 0.0,
				            -fit_x, 1.0, 0.0, 1.0,
			            ]),
		            }
	            }
            }
        )*
    };
}

impl_trans4_for!(Mat4, DMat4);

use common::field::*;
use common::utils::{TwoAdicSlice, unpack};
use itertools::Itertools;
use p3_util::log2_strict_usize;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::{PackedExtPoly, Point, Poly};

pub fn compress_hi_reference<F: Field, Ext: ExtensionField<F>>(
    poly: &Poly<F>,
    point: &Point<Ext>,
) -> Poly<Ext> {
    assert!(point.k() <= poly.k());
    let eq: Poly<Ext> = point.eq(Ext::ONE);
    let size_inner = poly.len() / eq.len();
    let mut out = Ext::zero_vec(size_inner);

    poly.chunks(size_inner)
        .zip_eq(eq.iter())
        .for_each(|(chunk, &w)| {
            out.iter_mut()
                .zip_eq(chunk.iter())
                .for_each(|(acc, &f)| *acc += w * f);
        });
    out.into()
}

pub fn compress_lo_reference<F: Field, Ext: ExtensionField<F>>(
    poly: &Poly<F>,
    point: &Point<Ext>,
) -> Poly<Ext> {
    assert!(point.k() <= poly.k());
    let eq: Poly<Ext> = point.eq(Ext::ONE);
    let chunk_size = eq.len();

    poly.chunks(chunk_size)
        .map(|chunk| {
            chunk
                .iter()
                .zip_eq(eq.iter())
                .map(|(&f, &w)| w * f)
                .sum::<Ext>()
        })
        .collect::<Vec<_>>()
        .into()
}

#[derive(Debug, Clone)]
pub enum MaybePacked<F: Field, Ext: ExtensionField<F>> {
    Unpacked(Poly<Ext>),
    Packed(PackedExtPoly<F, Ext>),
}

impl<F: Field, Ext: ExtensionField<F>> MaybePacked<F, Ext> {
    pub fn new_unpacked(point: &Point<Ext>) -> Self {
        MaybePacked::Unpacked(point.eq(Ext::ONE))
    }

    pub fn new(point: &Point<Ext>) -> Self {
        if SplitEq::<F, Ext>::can_pack(point.k()) {
            MaybePacked::Packed(point.eq_packed(Ext::ONE))
        } else {
            MaybePacked::Unpacked(point.eq(Ext::ONE))
        }
    }

    pub fn k(&self) -> usize {
        match self {
            MaybePacked::Unpacked(poly) => poly.k(),
            MaybePacked::Packed(poly) => poly.k() + log2_strict_usize(F::Packing::WIDTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SplitEq<F: Field, Ext: ExtensionField<F>> {
    eq0: MaybePacked<F, Ext>,
    eq1: Poly<Ext>,
}

impl<F: Field, Ext: ExtensionField<F>> SplitEq<F, Ext> {
    pub fn can_pack(k: usize) -> bool {
        k >= log2_strict_usize(F::Packing::WIDTH)
    }

    pub fn new_unpacked(point: &Point<Ext>) -> Self {
        let (z0, z1) = point.split_at(point.k() / 2);
        let eq0 = MaybePacked::new_unpacked(&z0);
        let eq1 = z1.eq(Ext::ONE);
        SplitEq { eq0, eq1 }
    }

    pub fn new(point: &Point<Ext>, scale: Ext) -> Self {
        let (z0, z1) = point.split_at(point.k() / 2);
        let eq0 = MaybePacked::new(&z0);
        let eq1 = z1.eq(scale);
        SplitEq { eq0, eq1 }
    }

    pub fn k(&self) -> usize {
        self.eq0.k() + self.eq1.k()
    }

    pub fn eval_base(&self, poly: &Poly<F>) -> Ext {
        assert_eq!(poly.k(), self.k());

        if let Some(constant) = (poly.len() == 1).then_some(*poly.first().unwrap()) {
            return constant.into();
        }

        match &self.eq0 {
            MaybePacked::Unpacked(eq0) => {
                if poly.k() < 15 {
                    poly.chunks(eq0.len())
                        .zip_eq(self.eq1.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq0.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<Ext>()
                                * w1
                        })
                        .sum::<Ext>()
                } else {
                    poly.par_chunks(eq0.len())
                        .zip_eq(self.eq1.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq0.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<Ext>()
                                * w1
                        })
                        .sum::<Ext>()
                }
            }

            MaybePacked::Packed(eq0) => {
                if poly.k() < 17 {
                    let poly = F::Packing::pack_slice(poly);
                    let sum = poly
                        .chunks(eq0.len())
                        .zip_eq(self.eq1.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq0.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<Ext::ExtensionPacking>()
                                * w1
                        })
                        .sum::<Ext::ExtensionPacking>();
                    unpack(&[sum]).into_iter().sum::<Ext>()
                } else {
                    let poly = F::Packing::pack_slice(poly);
                    let sum = poly
                        .par_chunks(eq0.len())
                        .zip_eq(self.eq1.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq0.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<Ext::ExtensionPacking>()
                                * w1
                        })
                        .sum::<Ext::ExtensionPacking>();
                    unpack(&[sum]).into_iter().sum::<Ext>()
                }
            }
        }
    }

    pub fn eval_ext(&self, poly: &Poly<Ext>) -> Ext {
        assert_eq!(poly.k(), self.k());

        if let Some(constant) = (poly.len() == 1).then_some(*poly.first().unwrap()) {
            return constant;
        }

        match &self.eq0 {
            MaybePacked::Unpacked(eq0) => {
                if poly.k() < 15 {
                    poly.chunks(eq0.len())
                        .zip_eq(self.eq1.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq0.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<Ext>()
                                * w1
                        })
                        .sum::<Ext>()
                } else {
                    poly.par_chunks(eq0.len())
                        .zip_eq(self.eq1.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq0.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<Ext>()
                                * w1
                        })
                        .sum::<Ext>()
                }
            }
            MaybePacked::Packed(eq0) => {
                if poly.k() < 15 {
                    let sum = poly
                        .chunks(eq0.len() * F::Packing::WIDTH)
                        .zip_eq(self.eq1.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .chunks(F::Packing::WIDTH)
                                .zip_eq(eq0.iter())
                                .map(|(chunk, &w0)| {
                                    Ext::ExtensionPacking::from_ext_slice(chunk) * w0
                                })
                                .sum::<Ext::ExtensionPacking>()
                                * w1
                        })
                        .sum::<Ext::ExtensionPacking>();
                    unpack(&[sum]).into_iter().sum::<Ext>()
                } else {
                    let sum = poly
                        .par_chunks(eq0.len() * F::Packing::WIDTH)
                        .zip_eq(self.eq1.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .chunks(F::Packing::WIDTH)
                                .zip_eq(eq0.iter())
                                .map(|(chunk, &w0)| {
                                    Ext::ExtensionPacking::from_ext_slice(chunk) * w0
                                })
                                .sum::<Ext::ExtensionPacking>()
                                * w1
                        })
                        .sum::<Ext::ExtensionPacking>();
                    unpack(&[sum]).into_iter().sum::<Ext>()
                }
            }
        }
    }

    pub fn eval_packed(&self, poly: &Poly<Ext::ExtensionPacking>) -> Ext {
        assert_eq!(poly.k() + log2_strict_usize(F::Packing::WIDTH), self.k());
        match &self.eq0 {
            MaybePacked::Packed(eq0) => {
                if poly.k() < 15 {
                    let sum = poly
                        .chunks(eq0.len())
                        .zip_eq(self.eq1.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq0.iter())
                                .map(|(&f, &w0)| f * w0)
                                .sum::<Ext::ExtensionPacking>()
                                * w1
                        })
                        .sum::<Ext::ExtensionPacking>();
                    unpack(&[sum]).into_iter().sum::<Ext>()
                } else {
                    let sum = poly
                        .par_chunks(eq0.len())
                        .zip_eq(self.eq1.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq0.iter())
                                .map(|(&f, &w0)| f * w0)
                                .sum::<Ext::ExtensionPacking>()
                                * w1
                        })
                        .sum::<Ext::ExtensionPacking>();
                    unpack(&[sum]).into_iter().sum::<Ext>()
                }
            }
            _ => unreachable!(),
        }
    }

    pub fn compress_hi(&self, poly: &Poly<F>) -> Poly<Ext> {
        assert!(self.k() <= poly.k());

        match &self.eq0 {
            MaybePacked::Unpacked(eq0) => {
                let size_outer = poly.len() / self.eq1.len();
                let size_inner = size_outer / eq0.len();
                if poly.k() < 15 {
                    let mut out = Ext::zero_vec(size_inner);
                    poly.chunks(size_outer)
                        .zip(self.eq1.iter())
                        .for_each(|(block, &w1)| {
                            block
                                .chunks(size_inner)
                                .zip_eq(eq0.iter())
                                .for_each(|(chunk, &w0)| {
                                    let w = w0 * w1;
                                    out.iter_mut()
                                        .zip_eq(chunk.iter())
                                        .for_each(|(acc, &f)| *acc += w * f);
                                });
                        });
                    out.into()
                } else {
                    let size_outer = poly.len() / self.eq1.len();
                    let size_inner = size_outer / eq0.len();
                    poly.par_chunks(size_outer)
                        .zip_eq(self.eq1.par_iter())
                        .map(|(block, &w1)| {
                            let mut out = Ext::zero_vec(size_inner);
                            block
                                .chunks(size_inner)
                                .zip_eq(eq0.iter())
                                .for_each(|(chunk, &w0)| {
                                    let w = w0 * w1;
                                    out.iter_mut()
                                        .zip_eq(chunk.iter())
                                        .for_each(|(acc, &f)| *acc += w * f);
                                });
                            out
                        })
                        .reduce(
                            || Ext::zero_vec(size_inner),
                            |mut a, b| {
                                a.iter_mut()
                                    .zip_eq(b.iter())
                                    .for_each(|(acc, &x)| *acc += x);
                                a
                            },
                        )
                        .into()
                }
            }
            MaybePacked::Packed(eq0) => {
                let size_outer = poly.len() / self.eq1.len();
                let size_inner = size_outer / (eq0.len() * F::Packing::WIDTH);
                if poly.k() < 15 {
                    let mut out = Ext::zero_vec(size_inner);
                    poly.chunks(size_outer)
                        .zip(self.eq1.iter())
                        .for_each(|(block, &w1)| {
                            block
                                .chunks(size_inner * F::Packing::WIDTH)
                                .zip_eq(eq0.iter())
                                .for_each(|(chunk, &w0)| {
                                    out.iter_mut().enumerate().for_each(|(i, acc)| {
                                        let packed =
                                            F::Packing::from_fn(|j| chunk[j * size_inner + i]);
                                        *acc += Ext::ExtensionPacking::to_ext_iter([w0 * packed])
                                            .sum::<Ext>()
                                            * w1;
                                    });
                                });
                        });
                    out.into()
                } else {
                    poly.par_chunks(size_outer)
                        .zip_eq(self.eq1.par_iter())
                        .map(|(block, &w1)| {
                            let mut out = Ext::zero_vec(size_inner);

                            block
                                .chunks(size_inner * F::Packing::WIDTH)
                                .zip_eq(eq0.iter())
                                .for_each(|(chunk, &w0)| {
                                    out.iter_mut().enumerate().for_each(|(i, acc)| {
                                        let packed =
                                            F::Packing::from_fn(|j| chunk[j * size_inner + i]);
                                        *acc += Ext::ExtensionPacking::to_ext_iter([w0 * packed])
                                            .sum::<Ext>()
                                            * w1;
                                    });
                                });
                            out
                        })
                        .reduce(
                            || Ext::zero_vec(size_inner),
                            |mut a, b| {
                                a.iter_mut()
                                    .zip_eq(b.iter())
                                    .for_each(|(acc, &x)| *acc += x);
                                a
                            },
                        )
                        .into()
                }
            }
        }
    }

    pub fn compress_lo(&self, poly: &Poly<F>) -> Poly<Ext> {
        assert!(self.k() <= poly.k());

        match &self.eq0 {
            MaybePacked::Unpacked(eq0) => {
                if poly.k() < 15 {
                    let size_outer = 1 << self.k();
                    poly.chunks(size_outer)
                        .map(|chunk| {
                            chunk
                                .chunks(eq0.len())
                                .zip_eq(self.eq1.iter())
                                .map(|(chunk, &w1)| {
                                    chunk
                                        .iter()
                                        .zip_eq(eq0.iter())
                                        .map(|(&f, &w0)| w0 * f)
                                        .sum::<Ext>()
                                        * w1
                                })
                                .sum::<Ext>()
                        })
                        .collect::<Vec<_>>()
                        .into()
                } else {
                    poly.par_chunks(1 << self.k())
                        .map(|chunk| {
                            chunk
                                .chunks(eq0.len())
                                .zip_eq(self.eq1.iter())
                                .map(|(chunk, &w1)| {
                                    chunk
                                        .iter()
                                        .zip_eq(eq0.iter())
                                        .map(|(&f, &w0)| w0 * f)
                                        .sum::<Ext>()
                                        * w1
                                })
                                .sum::<Ext>()
                        })
                        .collect::<Vec<_>>()
                        .into()
                }
            }
            MaybePacked::Packed(eq0) => {
                if poly.k() < 15 {
                    let size_outer = 1 << self.k();
                    poly.chunks(size_outer)
                        .map(|chunk| {
                            let chunk = F::Packing::pack_slice(chunk);
                            chunk
                                .chunks(eq0.len())
                                .zip_eq(self.eq1.iter())
                                .map(|(chunk, &w1)| {
                                    chunk
                                        .iter()
                                        .zip_eq(eq0.iter())
                                        .map(|(&f, &w0)| w0 * f)
                                        .sum::<Ext::ExtensionPacking>()
                                        * w1
                                })
                                .sum::<Ext::ExtensionPacking>()
                        })
                        .map(|packed| unpack(&[packed]).into_iter().sum::<Ext>())
                        .collect::<Vec<_>>()
                        .into()
                } else {
                    poly.par_chunks(1 << self.k())
                        .map(|chunk| {
                            let chunk = F::Packing::pack_slice(chunk);
                            let packed = chunk
                                .chunks(eq0.len())
                                .zip_eq(self.eq1.iter())
                                .map(|(chunk, &w1)| {
                                    chunk
                                        .iter()
                                        .zip_eq(eq0.iter())
                                        .map(|(&f, &w0)| w0 * f)
                                        .sum::<Ext::ExtensionPacking>()
                                        * w1
                                })
                                .sum::<Ext::ExtensionPacking>();
                            unpack(&[packed]).into_iter().sum::<Ext>()
                        })
                        .collect::<Vec<_>>()
                        .into()
                }
            }
        }
    }

    pub fn combine_into(&self, out: &mut [Ext], alpha: Option<Ext>) {
        assert_eq!(out.k(), self.k());
        match &self.eq0 {
            MaybePacked::Unpacked(eq0) => {
                if self.k() < 15 {
                    out.chunks_mut(eq0.len())
                        .zip(self.eq1.iter())
                        .for_each(|(chunk, &eq1)| match alpha {
                            Some(scale) => chunk
                                .iter_mut()
                                .zip(eq0.iter())
                                .for_each(|(out, &eq0)| *out += eq0 * eq1 * scale),
                            None => chunk
                                .iter_mut()
                                .zip(eq0.iter())
                                .for_each(|(out, &eq0)| *out += eq0 * eq1),
                        });
                } else {
                    out.par_chunks_mut(eq0.len())
                        .zip(self.eq1.par_iter())
                        .for_each(|(chunk, &eq1)| match alpha {
                            Some(scale) => chunk
                                .iter_mut()
                                .zip(eq0.iter())
                                .for_each(|(out, &eq0)| *out += eq0 * eq1 * scale),
                            None => chunk
                                .iter_mut()
                                .zip(eq0.iter())
                                .for_each(|(out, &eq0)| *out += eq0 * eq1),
                        });
                }
            }
            MaybePacked::Packed(eq0) => {
                if self.k() < 15 {
                    out.chunks_mut(eq0.len() * F::Packing::WIDTH)
                        .zip(self.eq1.iter())
                        .for_each(|(chunk, &eq1)| {
                            chunk
                                .chunks_mut(F::Packing::WIDTH)
                                .zip(eq0.iter())
                                .for_each(|(out, &eq0)| {
                                    let eqs = Ext::ExtensionPacking::to_ext_iter([eq0])
                                        .collect::<Vec<_>>();
                                    match alpha {
                                        Some(scale) => out
                                            .iter_mut()
                                            .zip_eq(eqs.iter())
                                            .for_each(|(out, &eq0)| *out += eq0 * eq1 * scale),
                                        None => out
                                            .iter_mut()
                                            .zip_eq(eqs.iter())
                                            .for_each(|(out, &eq0)| *out += eq0 * eq1),
                                    }
                                });
                        });
                } else {
                    out.par_chunks_mut(eq0.len() * F::Packing::WIDTH)
                        .zip(self.eq1.par_iter())
                        .for_each(|(chunk, &eq1)| {
                            chunk
                                .chunks_mut(F::Packing::WIDTH)
                                .zip(eq0.iter())
                                .for_each(|(out, &eq0)| {
                                    let eqs = Ext::ExtensionPacking::to_ext_iter([eq0])
                                        .collect::<Vec<_>>();
                                    match alpha {
                                        Some(scale) => out
                                            .iter_mut()
                                            .zip_eq(eqs.iter())
                                            .for_each(|(out, &eq0)| *out += eq0 * eq1 * scale),
                                        None => out
                                            .iter_mut()
                                            .zip_eq(eqs.iter())
                                            .for_each(|(out, &eq0)| *out += eq0 * eq1),
                                    }
                                });
                        });
                }
            }
        }
    }

    pub fn combine_into_packed(&self, out: &mut [Ext::ExtensionPacking], alpha: Ext) {
        assert_eq!(out.k() + log2_strict_usize(F::Packing::WIDTH), self.k());
        match &self.eq0 {
            MaybePacked::Packed(eq0) => {
                if self.k() < 15 {
                    out.chunks_mut(eq0.len())
                        .zip(self.eq1.iter())
                        .for_each(|(chunk, &eq1)| {
                            chunk
                                .iter_mut()
                                .zip(eq0.iter())
                                .for_each(|(out, &eq0)| *out += eq0 * eq1 * alpha);
                        });
                } else {
                    out.par_chunks_mut(eq0.len())
                        .zip(self.eq1.par_iter())
                        .for_each(|(chunk, &eq1)| {
                            chunk
                                .iter_mut()
                                .zip(eq0.iter())
                                .for_each(|(out, &eq0)| *out += eq0 * eq1 * alpha);
                        });
                }
            }
            MaybePacked::Unpacked(_) => unreachable!(),
        }
    }
}

// fast when poly_k is close to point k
// pub fn combine(&self, poly: &Poly<F>) -> Poly<Ext> {
//     assert!(self.k() <= poly.k());

//     match &self.eq0 {
//         MaybePacked::Unpacked(eq0) => {
//             let size_outer = 1 << self.k();
//             poly.chunks(size_outer)
//                 .map(|chunk| {
//                     chunk
//                         .par_chunks(eq0.len())
//                         .zip_eq(self.eq1.par_iter())
//                         .map(|(chunk, &w1)| {
//                             chunk
//                                 .iter()
//                                 .zip_eq(eq0.iter())
//                                 .map(|(&f, &w0)| w0 * f)
//                                 .sum::<Ext>()
//                                 * w1
//                         })
//                         .sum::<Ext>()
//                 })
//                 .collect::<Vec<_>>()
//                 .into()
//         }
//         MaybePacked::Packed(eq0) => {
//             let size_outer = 1 << self.k();
//             poly.chunks(size_outer)
//                 .map(|chunk| {
//                     let chunk = F::Packing::pack_slice(chunk);
//                     let sum = chunk
//                         .par_chunks(eq0.len())
//                         .zip_eq(self.eq1.par_iter())
//                         .map(|(chunk, &w1)| {
//                             chunk
//                                 .iter()
//                                 .zip_eq(eq0.iter())
//                                 .map(|(&f, &w0)| w0 * f)
//                                 .sum::<Ext::ExtensionPacking>()
//                                 * w1
//                         })
//                         .sum::<Ext::ExtensionPacking>();
//                     unpack(&[sum]).into_iter().sum::<Ext>()
//                 })
//                 .collect::<Vec<_>>()
//                 .into()
//         }
//     }
// }

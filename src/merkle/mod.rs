use crate::Error;
use digest::FixedOutputReset;
use p3_field::Field;
use sha2::Digest;
use std::fmt::Debug;

pub mod matrix;

pub trait Compress<T, const N: usize>: Send + Sync {
    fn compress(&self, input: [T; N]) -> T;
}

pub trait Hasher<Item, Out>: Send + Sync {
    fn hash(&self, input: &[Item]) -> Out;
    fn hash_iter<'a, I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = &'a Item>,
        Item: 'a + Clone,
    {
        self.hash(&input.into_iter().cloned().collect::<Vec<_>>())
    }
}

#[derive(Debug, Clone, Default)]
pub struct RustCrypto<D: Digest> {
    _0: std::marker::PhantomData<D>,
}

impl<D: Digest + FixedOutputReset + Send + Sync, const N: usize> Compress<[u8; 32], N>
    for RustCrypto<D>
{
    fn compress(&self, input: [[u8; 32]; N]) -> [u8; 32] {
        let mut h = D::new();
        input.iter().for_each(|e| Digest::update(&mut h, e));
        Digest::finalize(h).to_vec().try_into().unwrap()
    }
}

impl<D: Digest + FixedOutputReset + Send + Sync, F: Field> Hasher<F, [u8; 32]> for RustCrypto<D> {
    fn hash(&self, input: &[F]) -> [u8; 32] {
        let mut h = D::new();
        input
            .iter()
            .for_each(|el| Digest::update(&mut h, bincode::serialize(el).unwrap()));
        Digest::finalize(h).to_vec().try_into().unwrap()
    }
}

#[allow(clippy::precedence)]
pub(super) fn to_interleaved_index(k: usize, index: usize) -> usize {
    ((index << 1) & (1 << k) - 1) | (index >> (k - 1))
}

pub fn verify_merkle_proof<C, Node>(
    c: &C,
    claim: Node,
    mut index: usize,
    leaf: Node,
    witness: &[Node],
) -> Result<(), Error>
where
    C: Compress<Node, 2>,
    Node: Copy + Clone + Send + Sync + Debug + Eq + PartialEq,
{
    assert!(index < 1 << witness.len());
    let found = witness.iter().fold(leaf, |acc, &w| {
        let acc = c.compress(if index & 1 == 1 { [w, acc] } else { [acc, w] });
        index >>= 1;
        acc
    });
    (claim == found).then_some(()).ok_or(Error::Verify)
}

pub struct MerkleTree<F, Digest, H: Hasher<F, Digest>, C: Compress<Digest, 2>> {
    pub(crate) h: H,
    pub(crate) c: C,
    pub(crate) _phantom: std::marker::PhantomData<(F, Digest)>,
}

impl<F, Digest, H, C> MerkleTree<F, Digest, H, C>
where
    F: Field,
    H: Hasher<F, Digest>,
    C: Compress<Digest, 2>,
{
    pub fn new(h: H, c: C) -> Self {
        Self {
            h,
            c,
            _phantom: std::marker::PhantomData,
        }
    }
}

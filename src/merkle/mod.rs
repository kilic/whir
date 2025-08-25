use crate::Error;
use digest::FixedOutputReset;
use p3_field::{ExtensionField, Field};
use p3_symmetric::{
    CryptographicHasher, CryptographicPermutation, PaddingFreeSponge, PseudoCompressionFunction,
    TruncatedPermutation,
};
use sha2::Digest;
use std::fmt::Debug;

pub mod matrix;

pub trait Compress<T, const N: usize>: Sync {
    fn compress(&self, input: [T; N]) -> T;
}

#[derive(Debug, Clone, Default)]
pub struct RustCryptoCompress<D: Digest> {
    _0: std::marker::PhantomData<D>,
}

impl<D: Digest + FixedOutputReset + Sync, const N: usize> Compress<[u8; 32], N>
    for RustCryptoCompress<D>
{
    fn compress(&self, input: [[u8; 32]; N]) -> [u8; 32] {
        let mut h = D::new();
        input.iter().for_each(|e| Digest::update(&mut h, e));
        Digest::finalize(h).to_vec().try_into().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct PoseidonCompress<F: Field, Perm, const N: usize, const CHUNK: usize, const WIDTH: usize>
{
    poseidon: TruncatedPermutation<Perm, N, CHUNK, WIDTH>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Perm, const N: usize, const CHUNK: usize, const WIDTH: usize>
    PoseidonCompress<F, Perm, N, CHUNK, WIDTH>
{
    pub fn new(perm: Perm) -> PoseidonCompress<F, Perm, N, CHUNK, WIDTH> {
        PoseidonCompress {
            poseidon: TruncatedPermutation::new(perm),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<
        F: Field,
        Perm: CryptographicPermutation<[F; WIDTH]>,
        const N: usize,
        const CHUNK: usize,
        const WIDTH: usize,
    > Compress<[F; CHUNK], N> for PoseidonCompress<F, Perm, N, CHUNK, WIDTH>
{
    fn compress(&self, input: [[F; CHUNK]; N]) -> [F; CHUNK] {
        self.poseidon.compress(input)
    }
}

pub trait Hasher<Item, Out>: Sync {
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
pub struct RustCryptoHasher<D: Digest> {
    _0: std::marker::PhantomData<D>,
}

impl<D: Digest + FixedOutputReset + Sync, F: Field> Hasher<F, [u8; 32]> for RustCryptoHasher<D> {
    fn hash(&self, input: &[F]) -> [u8; 32] {
        let mut h = D::new();
        input
            .iter()
            .for_each(|el| Digest::update(&mut h, bincode::serialize(el).unwrap()));
        Digest::finalize(h).to_vec().try_into().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct PoseidonHasher<
    F: Field,
    Perm: CryptographicPermutation<[F; WIDTH]>,
    const WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
> {
    poseidon: PaddingFreeSponge<Perm, WIDTH, RATE, OUT>,
    _marker: std::marker::PhantomData<F>,
}

impl<
        F: Field,
        Perm: CryptographicPermutation<[F; WIDTH]>,
        const WIDTH: usize,
        const RATE: usize,
        const OUT: usize,
    > PoseidonHasher<F, Perm, WIDTH, RATE, OUT>
{
    pub fn new(perm: Perm) -> PoseidonHasher<F, Perm, WIDTH, RATE, OUT> {
        PoseidonHasher {
            poseidon: PaddingFreeSponge::new(perm),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<
        F: Field,
        Perm: CryptographicPermutation<[F; WIDTH]>,
        const WIDTH: usize,
        const RATE: usize,
        const OUT: usize,
    > Hasher<F, [F; OUT]> for PoseidonHasher<F, Perm, WIDTH, RATE, OUT>
{
    fn hash(&self, input: &[F]) -> [F; OUT] {
        self.poseidon.hash_slice(input)
    }

    fn hash_iter<'a, I>(&self, input: I) -> [F; OUT]
    where
        I: IntoIterator<Item = &'a F>,
    {
        self.poseidon.hash_iter(input.into_iter().cloned())
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
    Node: Copy + Clone + Sync + Debug + Eq + PartialEq,
{
    assert!(index < 1 << witness.len());
    let found = witness.iter().fold(leaf, |acc, &w| {
        let acc = c.compress(if index & 1 == 1 { [w, acc] } else { [acc, w] });
        index >>= 1;
        acc
    });
    (claim == found).then_some(()).ok_or(Error::Verify)
}

pub struct MerkleTree<F, Ext, Digest, H: Hasher<F, Digest>, C: Compress<Digest, 2>> {
    pub(crate) h: H,
    pub(crate) c: C,
    pub(crate) _phantom: std::marker::PhantomData<(F, Ext, Digest)>,
}

impl<F, Ext, Digest, H, C> MerkleTree<F, Ext, Digest, H, C>
where
    F: Field,
    Ext: ExtensionField<F>,
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

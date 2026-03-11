#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Error {
    Transcript,
    Verify,
    IO,
}

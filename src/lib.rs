use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, PyResult, Python};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

// maybe go back to strsim
pub fn levenshtein_distance(seq1: &str, seq2: &str) -> usize {
    let size_x = seq1.len() + 1;
    let size_y = seq2.len() + 1;

    let mut matrix = vec![vec![0; size_y]; size_x];

    for x in 0..size_x {
        matrix[x][0] = x;
    }

    for y in 0..size_y {
        matrix[0][y] = y;
    }

    for x in 1..size_x {
        for y in 1..size_y {
            let substitution_cost =
                if seq1.chars().nth(x - 1).unwrap() == seq2.chars().nth(y - 1).unwrap() {
                    0
                } else {
                    1
                };

            matrix[x][y] = usize::min(
                matrix[x - 1][y] + 1,
                usize::min(
                    matrix[x - 1][y - 1] + substitution_cost,
                    matrix[x][y - 1] + 1,
                ),
            );
        }
    }

    matrix[size_x - 1][size_y - 1]
}

pub fn hamming_distance(seq1: &str, seq2: &str, max: Option<usize>, ignore_n: bool) -> usize {
    if seq1.len() != seq2.len() {
        panic!("Sequences are not the same length");
    }

    let max = max.unwrap_or(usize::MAX);

    let mut mismatches = 0;

    for (c1, c2) in seq1.chars().zip(seq2.chars()) {
        if ignore_n && (c1 == 'N' || c2 == 'N') {
            continue;
        }

        if c1 != c2 {
            mismatches += 1;

            if mismatches >= max {
                break;
            }
        }
    }

    mismatches
}

pub fn distance(s1: &str, s2: &str, try_both_distances: bool) -> usize {
    if s1.len() == s2.len() {
        let hamm_dist = hamming_distance(s1, s2, None, false);
        if hamm_dist > 2 && try_both_distances {
            let lev_dist = levenshtein_distance(s1, s2);
            hamm_dist.min(lev_dist)
        } else {
            hamm_dist
        }
    } else {
        levenshtein_distance(s1, s2)
    }
}

#[pyfunction]
fn pairwise_min_distance(
    strings: Vec<(usize, String, String)>,
) -> PyResult<Vec<(usize, usize, bool)>> {
    // return a vector of usizes with the index of the string with the minimum distance to each string, except itself

    let mut min_distances = vec![];

    for (i, forward_bc, reverse_bc) in strings.iter() {
        let mut min_distance = usize::MAX;
        let mut min_index = 0;
        let mut did_use_lev = false;

        if i % 5 == 0 {
            println!("Processing string {}", i);
        }

        let distances = strings
            .par_iter()
            .filter_map(|(j, fbc, rbc)| {
                if i != j {
                    let used_lev =
                        !(forward_bc.len() == fbc.len() || reverse_bc.len() == rbc.len());

                    let distance =
                        distance(forward_bc, fbc, true) + distance(reverse_bc, rbc, true);
                    Some((*j, distance, used_lev))
                } else {
                    None
                }
            })
            .collect::<Vec<(usize, usize, bool)>>();

        // determine the minimum distance and the index of the string with the minimum distance
        for (j, distance, used_lev) in distances.iter() {
            if *distance < min_distance {
                min_distance = *distance;
                min_index = *j;
                did_use_lev = *used_lev;
            }
        }

        min_distances.push((min_index, min_distance, did_use_lev));
    }

    Ok(min_distances)
}

/// A Python module implemented in Rust.
#[pymodule]
fn gsf_ims_fitness(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pairwise_min_distance, m)?)?;
    Ok(())
}

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

fn ensure_same_len3<T, U, V>(a: &[T], b: &[U], c: &[V]) -> PyResult<()> {
    if a.len() != b.len() || a.len() != c.len() {
        Err(PyValueError::new_err(
            "all input arrays must have the same length",
        ))
    } else {
        Ok(())
    }
}

fn ensure_same_len5<T, U, V, W, X>(a: &[T], b: &[U], c: &[V], d: &[W], e: &[X]) -> PyResult<()> {
    let n = a.len();
    if b.len() != n || c.len() != n || d.len() != n || e.len() != n {
        Err(PyValueError::new_err(
            "all input arrays must have the same length",
        ))
    } else {
        Ok(())
    }
}

fn build_block_slices(block_number: &[i64]) -> Vec<(usize, usize)> {
    if block_number.is_empty() {
        return Vec::new();
    }

    let mut slices = Vec::new();
    let mut start = 0usize;
    for i in 1..block_number.len() {
        if block_number[i] != block_number[i - 1] {
            slices.push((start, i));
            start = i;
        }
    }
    slices.push((start, block_number.len()));
    slices
}

fn event_code(event: &str) -> i8 {
    match event {
        "Swap_X2Y" => 1,
        "Swap_Y2X" => 2,
        "Mint" => 3,
        "Burn" => 4,
        _ => 0,
    }
}

fn is_close(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    diff <= 1e-8 + 1e-5 * b.abs()
}

fn upper_bound(values: &[i64], target: i64) -> usize {
    let mut left = 0usize;
    let mut right = values.len();
    while left < right {
        let mid = (left + right) / 2;
        if values[mid] <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn lower_bound(values: &[i64], target: i64) -> usize {
    let mut left = 0usize;
    let mut right = values.len();
    while left < right {
        let mid = (left + right) / 2;
        if values[mid] < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn median_nan(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = finite.len();
    if n % 2 == 1 {
        finite[n / 2]
    } else {
        (finite[n / 2 - 1] + finite[n / 2]) / 2.0
    }
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        cov_xy += dx * (y[i] - mean_y);
        var_x += dx * dx;
    }
    let slope = cov_xy / var_x;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

fn lo_cdf(v: f64) -> f64 {
    if !v.is_finite() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return 0.0;
    }
    let mut total = 1.0;
    for k in 1..=10_000 {
        let kv = (k as f64) * v;
        let term = 2.0 * (1.0 - 4.0 * kv * kv) * (-2.0 * kv * kv).exp();
        total += term;
        if term.abs() < 1e-12 {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

fn lo_modified_rs(x: &[f64], q: Option<usize>, alpha: f64) -> PyResult<(f64, f64, f64)> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / (n as f64);
    let xc: Vec<f64> = x.iter().map(|v| *v - mean).collect();

    let q_used = if let Some(qv) = q {
        if qv >= n {
            return Err(PyValueError::new_err("q must satisfy 0 <= q < len(series)."));
        }
        qv
    } else {
        let mean0 = x[..n - 1].iter().sum::<f64>() / ((n - 1) as f64);
        let mean1 = x[1..].iter().sum::<f64>() / ((n - 1) as f64);
        let mut dot01 = 0.0;
        let mut dot00 = 0.0;
        let mut dot11 = 0.0;
        for i in 0..(n - 1) {
            let x0 = x[i] - mean0;
            let x1 = x[i + 1] - mean1;
            dot01 += x0 * x1;
            dot00 += x0 * x0;
            dot11 += x1 * x1;
        }
        let denom = (dot00 * dot11).sqrt();
        let mut rho1 = if denom == 0.0 { 0.0 } else { dot01 / denom };
        rho1 = rho1.clamp(1e-3, 0.999);
        let q_auto = (((3.0 * (n as f64)) / 2.0).powf(1.0 / 3.0)
            * (2.0 * rho1 / (1.0 - rho1 * rho1)).powf(2.0 / 3.0))
            .abs() as usize;
        q_auto.min(n - 1).min(99)
    };

    let mut partial = 0.0;
    let mut min_partial = 0.0;
    let mut max_partial = 0.0;
    for val in &xc {
        partial += *val;
        if partial < min_partial {
            min_partial = partial;
        }
        if partial > max_partial {
            max_partial = partial;
        }
    }
    let r_n = max_partial - min_partial;

    let gamma0 = xc.iter().map(|v| v * v).sum::<f64>() / (n as f64);
    let mut s2_q = gamma0;
    for lag in 1..=q_used {
        let mut gamma_lag = 0.0;
        for i in lag..n {
            gamma_lag += xc[i] * xc[i - lag];
        }
        gamma_lag /= n as f64;
        let weight = 1.0 - (lag as f64) / ((q_used + 1) as f64);
        s2_q += 2.0 * weight * gamma_lag;
    }

    if s2_q <= 0.0 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let v_n_q = r_n / ((n as f64 * s2_q).sqrt());
    let cdf_value = lo_cdf(v_n_q);
    let p_lower = cdf_value;
    let p_upper = 1.0 - cdf_value;
    let p_two_sided = (2.0 * p_lower.min(p_upper)).min(1.0);

    match alpha {
        0.10 | 0.05 | 0.01 => Ok((p_two_sided, p_upper, p_lower)),
        _ => Err(PyValueError::new_err("alpha must be one of {0.10, 0.05, 0.01}.")),
    }
}

fn classical_rs_hurst(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 8 {
        return f64::NAN;
    }
    let max_scale = n / 2;
    if max_scale < 2 {
        return f64::NAN;
    }

    let max_power = (max_scale as f64).log2().floor() as usize;
    let mut valid_scales = Vec::<f64>::new();
    let mut mean_rs = Vec::<f64>::new();

    for p in 1..=max_power {
        let scale = 1usize << p;
        let n_blocks = n / scale;
        if n_blocks < 2 {
            continue;
        }

        let mut rs_vals = Vec::<f64>::new();
        for b in 0..n_blocks {
            let start = b * scale;
            let end = start + scale;
            let block = &x[start..end];
            let mean = block.iter().sum::<f64>() / (scale as f64);

            let mut centered = Vec::with_capacity(scale);
            let mut sigma_sq = 0.0;
            for v in block {
                let c = *v - mean;
                sigma_sq += c * c;
                centered.push(c);
            }
            let sigma = (sigma_sq / (scale as f64)).sqrt();
            if sigma <= 0.0 {
                continue;
            }

            let mut csum = 0.0;
            let mut min_csum = 0.0;
            let mut max_csum = 0.0;
            for c in centered {
                csum += c;
                if csum < min_csum {
                    min_csum = csum;
                }
                if csum > max_csum {
                    max_csum = csum;
                }
            }
            let rs = (max_csum - min_csum) / sigma;
            if rs.is_finite() && rs > 0.0 {
                rs_vals.push(rs);
            }
        }

        if rs_vals.len() >= 2 {
            valid_scales.push(scale as f64);
            mean_rs.push(rs_vals.iter().sum::<f64>() / (rs_vals.len() as f64));
        }
    }

    if valid_scales.len() < 2 {
        return f64::NAN;
    }

    let log_s: Vec<f64> = valid_scales.iter().map(|v| v.ln()).collect();
    let log_rs: Vec<f64> = mean_rs.iter().map(|v| v.ln()).collect();
    let (slope, _) = linear_regression(&log_s, &log_rs);
    slope
}

fn periodogram_hurst(x: &[f64], low_freq_frac: f64) -> PyResult<f64> {
    if !(low_freq_frac > 0.0 && low_freq_frac <= 1.0) {
        return Err(PyValueError::new_err(
            "low_freq_frac must satisfy 0 < low_freq_frac <= 1."
        ));
    }

    let n = x.len();
    let mean = x.iter().sum::<f64>() / (n as f64);
    let mut buffer: Vec<Complex<f64>> = x
        .iter()
        .map(|v| Complex { re: *v - mean, im: 0.0 })
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    let n_pos = n / 2;
    if n_pos < 10 {
        return Err(PyValueError::new_err(
            "Not enough positive Fourier frequencies for periodogram regression."
        ));
    }

    let n_fit = ((low_freq_frac * (n_pos as f64)).ceil() as usize).max(2).min(n_pos);
    let mut log_freq = Vec::<f64>::new();
    let mut log_peri = Vec::<f64>::new();

    for k in 1..=n_fit {
        let freq = (k as f64) / (n as f64);
        let peri = buffer[k].norm_sqr() / (n as f64);
        if peri > 0.0 {
            log_freq.push(freq.ln());
            log_peri.push(peri.ln());
        }
    }

    if log_freq.len() < 2 {
        return Ok(f64::NAN);
    }

    let (slope, _) = linear_regression(&log_freq, &log_peri);
    Ok((1.0 - slope) / 2.0)
}

fn dfa_scales(min_scale: usize, max_scale: usize, n_scales: usize) -> Vec<usize> {
    let log_min = (min_scale as f64).ln();
    let log_max = (max_scale as f64).ln();
    let mut scales = Vec::<usize>::new();
    for i in 0..n_scales {
        let scale = if i == 0 {
            min_scale
        } else if i + 1 == n_scales {
            max_scale
        } else {
            let t = if n_scales == 1 {
                0.0
            } else {
                (i as f64) / ((n_scales - 1) as f64)
            };
            (log_min + t * (log_max - log_min)).exp() as usize
        };
        if scale >= min_scale {
            if scales.last().copied() != Some(scale) {
                scales.push(scale);
            }
        }
    }
    scales
}

fn dfa_hurst(x: &[f64], min_scale: usize, max_scale_in: Option<usize>, n_scales: usize) -> PyResult<f64> {
    let n = x.len();
    let max_scale = if let Some(v) = max_scale_in {
        v
    } else {
        (n / 4).min(n - 1).max(min_scale + 1)
    };

    if min_scale < 2 {
        return Err(PyValueError::new_err("dfa_min_scale must be at least 2."));
    }
    if max_scale <= min_scale {
        return Err(PyValueError::new_err("dfa_max_scale must be larger than dfa_min_scale."));
    }
    if n_scales < 2 {
        return Err(PyValueError::new_err("dfa_n_scales must be at least 2."));
    }

    let mean = x.iter().sum::<f64>() / (n as f64);
    let mut profile = Vec::with_capacity(n);
    let mut csum = 0.0;
    for v in x {
        csum += *v - mean;
        profile.push(csum);
    }

    let scales = dfa_scales(min_scale, max_scale, n_scales);
    let mut valid_scales = Vec::<f64>::new();
    let mut fluctuations = Vec::<f64>::new();

    for &scale in &scales {
        let n_boxes = n / scale;
        if n_boxes < 2 {
            continue;
        }

        let t_mean = ((scale - 1) as f64) / 2.0;
        let mut t_var = 0.0;
        for t in 0..scale {
            let dt = (t as f64) - t_mean;
            t_var += dt * dt;
        }
        t_var /= scale as f64;

        let mut detr_sq_sum = 0.0;
        for b in 0..n_boxes {
            let start = b * scale;
            let end = start + scale;
            let block = &profile[start..end];
            let box_mean = block.iter().sum::<f64>() / (scale as f64);

            let mut cov = 0.0;
            for t in 0..scale {
                cov += ((t as f64) - t_mean) * (block[t] - box_mean);
            }
            cov /= scale as f64;
            let slope = cov / t_var;
            let intercept = box_mean - slope * t_mean;

            for t in 0..scale {
                let trend = intercept + slope * (t as f64);
                let detr = block[t] - trend;
                detr_sq_sum += detr * detr;
            }
        }

        let fluc = (detr_sq_sum / ((n_boxes * scale) as f64)).sqrt();
        if fluc.is_finite() && fluc > 0.0 {
            valid_scales.push(scale as f64);
            fluctuations.push(fluc);
        }
    }

    if valid_scales.len() < 2 {
        return Ok(f64::NAN);
    }

    let log_scales: Vec<f64> = valid_scales.iter().map(|v| v.ln()).collect();
    let log_fluc: Vec<f64> = fluctuations.iter().map(|v| v.ln()).collect();
    let (slope, _) = linear_regression(&log_scales, &log_fluc);
    Ok(slope)
}

fn ffill_nan_1d(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    let mut last_valid = None::<f64>;
    for v in &mut out {
        if v.is_nan() {
            if let Some(last) = last_valid {
                *v = last;
            }
        } else {
            last_valid = Some(*v);
        }
    }
    out
}

fn diff(values: &[f64]) -> Vec<f64> {
    if values.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len() - 1);
    for i in 1..values.len() {
        out.push(values[i] - values[i - 1]);
    }
    out
}

fn log_vec(values: &[f64]) -> Vec<f64> {
    values.iter().map(|v| v.ln()).collect()
}

fn sumsq_ignore_nan(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .map(|v| v * v)
        .sum::<f64>()
}

fn quantile_linear(values: &[f64], probs: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    probs
        .iter()
        .map(|p| {
            if n == 1 {
                return sorted[0];
            }
            let pos = *p * ((n - 1) as f64);
            let lower = pos.floor() as usize;
            let upper = pos.ceil() as usize;
            if lower == upper {
                sorted[lower]
            } else {
                let w = pos - (lower as f64);
                sorted[lower] * (1.0 - w) + sorted[upper] * w
            }
        })
        .collect()
}

fn resample_last(timestamp_ns: &[i64], price: &[f64], liq: &[f64], freq_min: i64) -> (Vec<f64>, Vec<f64>) {
    let bin_ns = freq_min * 60_000_000_000i64;
    let bin_id: Vec<i64> = timestamp_ns.iter().map(|v| *v / bin_ns).collect();

    let mut last_pos = Vec::<usize>::new();
    for i in 1..bin_id.len() {
        if bin_id[i] != bin_id[i - 1] {
            last_pos.push(i - 1);
        }
    }
    last_pos.push(bin_id.len() - 1);

    let present_bins: Vec<i64> = last_pos.iter().map(|&i| bin_id[i]).collect();
    let first_bin = present_bins[0];
    let n_bins = (present_bins[present_bins.len() - 1] - first_bin + 1) as usize;

    let mut price_last = vec![f64::NAN; n_bins];
    let mut liq_last = vec![f64::NAN; n_bins];
    for (&pos, &bin) in last_pos.iter().zip(present_bins.iter()) {
        let where_idx = (bin - first_bin) as usize;
        price_last[where_idx] = price[pos];
        liq_last[where_idx] = liq[pos];
    }
    (price_last, liq_last)
}

fn acf_fft(series: &[f64], lags: usize) -> Vec<f64> {
    let n = series.len();
    let mean = series.iter().sum::<f64>() / (n as f64);
    let m = (2 * n).next_power_of_two();
    let mut buffer = vec![Complex { re: 0.0, im: 0.0 }; m];
    for i in 0..n {
        buffer[i].re = series[i] - mean;
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(m);
    let ifft = planner.plan_fft_inverse(m);
    fft.process(&mut buffer);
    for v in &mut buffer {
        *v = *v * v.conj();
    }
    ifft.process(&mut buffer);

    let scale = (m as f64) * (n as f64);
    let acov0 = buffer[0].re / scale;
    let max_lag = lags.min(n - 1);
    let mut out = vec![0.0; max_lag + 1];
    for k in 0..=max_lag {
        let acov = buffer[k].re / scale;
        out[k] = acov / acov0;
    }
    out
}

fn ensure_same_len2<T, U>(a: &[T], b: &[U]) -> PyResult<()> {
    if a.len() != b.len() {
        Err(PyValueError::new_err(
            "all input arrays must have the same length",
        ))
    } else {
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature = (block_number, wallet, event, verbose=false))]
fn find_sandwich(
    block_number: Vec<i64>,
    wallet: Vec<String>,
    event: Vec<String>,
    verbose: bool,
) -> PyResult<Vec<String>> {
    let _ = verbose;
    ensure_same_len3(&block_number, &wallet, &event)?;
    if block_number.is_empty() {
        return Ok(Vec::new());
    }

    let event_code_vec: Vec<i8> = event.iter().map(|e| event_code(e)).collect();
    let mut labels = vec!["Not".to_string(); event.len()];

    for (block_start, block_end) in build_block_slices(&block_number) {
        let mut swap_loc = Vec::new();
        for idx in block_start..block_end {
            let code = event_code_vec[idx];
            if code == 1 || code == 2 {
                swap_loc.push(idx - block_start);
            }
        }
        if swap_loc.len() < 3 {
            continue;
        }

        let mut swap_codes = Vec::with_capacity(swap_loc.len());
        let mut swap_wallet = Vec::with_capacity(swap_loc.len());
        for &loc in &swap_loc {
            swap_codes.push(event_code_vec[block_start + loc]);
            swap_wallet.push(&wallet[block_start + loc]);
        }

        let n_swaps = swap_codes.len();
        let mut segments = Vec::<(usize, usize, usize)>::new();
        let mut n = 0usize;
        while n + 3 <= n_swaps {
            let front_code = swap_codes[n];
            let mut m = 0usize;
            while n + m < n_swaps - 1 && swap_codes[n + m + 1] == front_code {
                m += 1;
            }

            if m == 0 {
                n += 1;
                continue;
            }
            if n + m == n_swaps - 1 {
                n += m + 1;
                continue;
            }

            let back_wallet = swap_wallet[n + m + 1];
            while swap_wallet[n] != back_wallet && m > 0 {
                n += 1;
                m -= 1;
            }
            if m == 0 {
                continue;
            }

            segments.push((n, m, n + m + 1));
            n += m + 2;
        }

        for (front_s, _m, back_s) in segments {
            let seg_swap_idx: Vec<usize> = (front_s..=back_s).collect();
            let seg_event_loc: Vec<usize> = seg_swap_idx.iter().map(|&idx| swap_loc[idx]).collect();

            let front_ev = seg_event_loc[0];
            let victim_ev = if seg_event_loc.len() > 2 {
                seg_event_loc[1..seg_event_loc.len() - 1].to_vec()
            } else {
                Vec::new()
            };
            let back_ev = *seg_event_loc.last().unwrap();

            let attacker = swap_wallet[front_s];
            let mut is_self = true;
            for &idx in &seg_swap_idx {
                if swap_wallet[idx] != attacker {
                    is_self = false;
                    break;
                }
            }

            if is_self {
                labels[block_start + front_ev] = "Front_Self".to_string();
                for &loc in &victim_ev {
                    labels[block_start + loc] = "Victim_Self".to_string();
                }
                labels[block_start + back_ev] = "Back_Self".to_string();
                continue;
            }

            let mut mint_cand = Vec::new();
            let mut burn_cand = Vec::new();
            for loc in (front_ev + 1)..back_ev {
                let idx = block_start + loc;
                if wallet[idx] != *attacker {
                    continue;
                }
                match event_code_vec[idx] {
                    3 => mint_cand.push(loc),
                    4 => burn_cand.push(loc),
                    _ => {}
                }
            }

            let mut mint_ev = None;
            let mut burn_ev = None;
            if !victim_ev.is_empty() && !mint_cand.is_empty() && !burn_cand.is_empty() {
                let first_victim = victim_ev[0];
                let last_victim = *victim_ev.last().unwrap();
                let mint_before = mint_cand.iter().copied().filter(|&x| x < first_victim).last();
                let burn_after = burn_cand.iter().copied().find(|&x| x > last_victim);
                if let (Some(mint_candidate), Some(burn_candidate)) = (mint_before, burn_after) {
                    if mint_candidate < burn_candidate {
                        mint_ev = Some(mint_candidate);
                        burn_ev = Some(burn_candidate);
                    }
                }
            }

            if let (Some(mint_idx), Some(burn_idx)) = (mint_ev, burn_ev) {
                labels[block_start + front_ev] = "Front_Mix".to_string();
                labels[block_start + mint_idx] = "Mint_Mix".to_string();
                for &loc in &victim_ev {
                    labels[block_start + loc] = "Victim_Mix".to_string();
                }
                labels[block_start + burn_idx] = "Burn_Mix".to_string();
                labels[block_start + back_ev] = "Back_Mix".to_string();
            } else {
                labels[block_start + front_ev] = "Front".to_string();
                for &loc in &victim_ev {
                    labels[block_start + loc] = "Victim".to_string();
                }
                labels[block_start + back_ev] = "Back".to_string();
            }
        }
    }

    Ok(labels)
}

#[pyfunction]
#[pyo3(signature = (block_number, wallet, event, tx_hash=None))]
fn find_echo(
    block_number: Vec<i64>,
    wallet: Vec<String>,
    event: Vec<String>,
    tx_hash: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    ensure_same_len3(&block_number, &wallet, &event)?;
    if let Some(ref tx) = tx_hash {
        if tx.len() != block_number.len() {
            return Err(PyValueError::new_err(
                "tx_hash must have the same length as the other inputs",
            ));
        }
    }
    if block_number.is_empty() {
        return Ok(Vec::new());
    }

    let n = block_number.len();
    let mut pair_match = vec![false; n.saturating_sub(1)];
    for i in 0..n.saturating_sub(1) {
        pair_match[i] =
            wallet[i + 1] == wallet[i] && event[i + 1] != event[i] && block_number[i + 1] == block_number[i];
    }

    let mut codes = vec![0u8; n];
    if n >= 3 {
        for i in 1..(n - 1) {
            if pair_match[i - 1] {
                codes[i] = 1;
            }
            if pair_match[i] {
                codes[i] = 2;
            }
            if pair_match[i - 1] && pair_match[i] {
                codes[i] = 3;
            }
        }
    }

    let mut out: Vec<String> = if tx_hash.is_some() {
        codes
            .iter()
            .map(|code| match code {
                0 => "Not",
                1 => "Echo_Start",
                2 => "Echo_End",
                _ => "Echo",
            })
            .map(str::to_string)
            .collect()
    } else {
        codes
            .iter()
            .map(|code| match code {
                0 => "Not",
                1 => "Echo_Start",
                2 => "Echo",
                _ => "Echo_End",
            })
            .map(str::to_string)
            .collect()
    };

    if let Some(tx) = tx_hash {
        let mut pair_match_tx = vec![false; n.saturating_sub(1)];
        for i in 0..n.saturating_sub(1) {
            pair_match_tx[i] = pair_match[i] && tx[i + 1] == tx[i];
        }
        if n >= 3 {
            for i in 1..(n - 1) {
                if pair_match_tx[i - 1] || pair_match_tx[i] {
                    out[i].push_str("_1_Tx");
                }
            }
        }
    }

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (event, block_number, log_index, tick_upper, tick_lower, amount, verbose=false))]
fn find_jit(
    event: Vec<String>,
    block_number: Vec<i64>,
    log_index: Vec<i64>,
    tick_upper: Vec<f64>,
    tick_lower: Vec<f64>,
    amount: Vec<f64>,
    verbose: bool,
) -> PyResult<Vec<u8>> {
    let _ = verbose;
    ensure_same_len5(&event, &block_number, &log_index, &tick_upper, &tick_lower)?;
    if amount.len() != event.len() {
        return Err(PyValueError::new_err(
            "all input arrays must have the same length",
        ));
    }
    let n = event.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut jit_flag = vec![0u8; n];
    for (block_start, block_end) in build_block_slices(&block_number) {
        if block_end - block_start < 3 {
            continue;
        }

        let mut mint_pos = Vec::new();
        let mut burn_pos = Vec::new();
        let mut swap_pos = Vec::new();
        for loc in 0..(block_end - block_start) {
            match event[block_start + loc].as_str() {
                "Mint" => mint_pos.push(loc),
                "Burn" => burn_pos.push(loc),
                "Swap_X2Y" | "Swap_Y2X" => swap_pos.push(loc),
                _ => {}
            }
        }
        if mint_pos.is_empty() || burn_pos.is_empty() || swap_pos.is_empty() {
            continue;
        }

        let swap_logs: Vec<i64> = swap_pos.iter().map(|&pos| log_index[block_start + pos]).collect();
        for &mp in &mint_pos {
            let m_log = log_index[block_start + mp];
            let mut mint_is_jit = false;

            for &bp in &burn_pos {
                let b_log = log_index[block_start + bp];
                if b_log <= m_log {
                    continue;
                }
                if !is_close(tick_upper[block_start + bp], tick_upper[block_start + mp])
                    || !is_close(tick_lower[block_start + bp], tick_lower[block_start + mp])
                    || !is_close(amount[block_start + bp], amount[block_start + mp])
                {
                    continue;
                }

                let left = upper_bound(&swap_logs, m_log);
                let right = lower_bound(&swap_logs, b_log);
                if left < right {
                    mint_is_jit = true;
                    jit_flag[block_start + bp] = 1;
                    for &sp in &swap_pos[left..right] {
                        jit_flag[block_start + sp] = 1;
                    }
                }
            }

            if mint_is_jit {
                jit_flag[block_start + mp] = 1;
            }
        }
    }

    Ok(jit_flag)
}

#[pyfunction]
fn liq_change(
    pre_event: Vec<String>,
    pre_liquidity: Vec<f64>,
    pre_tick: Vec<f64>,
    pre_amount: Vec<f64>,
    event: Vec<String>,
    liquidity: Vec<f64>,
    amount: Vec<f64>,
    tick: Vec<f64>,
    tick_upper: Vec<f64>,
    tick_lower: Vec<f64>,
) -> PyResult<Vec<u8>> {
    ensure_same_len5(&pre_event, &pre_liquidity, &pre_tick, &pre_amount, &pre_amount)?;
    ensure_same_len5(&event, &liquidity, &amount, &tick, &tick_upper)?;
    if tick_lower.len() != event.len() {
        return Err(PyValueError::new_err(
            "all input arrays must have the same length",
        ));
    }

    let mut last_liq;
    let mut curr_tick;
    let n_pre = pre_event.len();

    if n_pre > 0 {
        let mut last_liq_opt = None;
        let mut curr_tick_opt = None;
        for idx in (0..n_pre).rev() {
            let liq = pre_liquidity[idx];
            if !liq.is_nan() {
                last_liq_opt = Some(liq);
                break;
            }
        }
        for idx in (0..n_pre).rev() {
            let tk = pre_tick[idx];
            if !tk.is_nan() {
                curr_tick_opt = Some(tk);
                break;
            }
        }

        if let Some(v) = last_liq_opt {
            last_liq = v;
            curr_tick = curr_tick_opt.unwrap_or(0.0);
        } else {
            last_liq = 0.0;
            curr_tick = tick.iter().copied().find(|v| !v.is_nan()).unwrap_or(0.0);
        }

        let mut flag = n_pre as isize - 1;
        while flag >= 0 {
            match pre_event[flag as usize].as_str() {
                "Mint" => {
                    last_liq += pre_amount[flag as usize];
                    flag -= 1;
                }
                "Burn" => {
                    last_liq -= pre_amount[flag as usize];
                    flag -= 1;
                }
                _ => break,
            }
        }
    } else {
        last_liq = 0.0;
        curr_tick = tick.iter().copied().find(|v| !v.is_nan()).unwrap_or(0.0);
    }

    let mut active_change = Vec::with_capacity(event.len());
    for i in 0..event.len() {
        match event[i].as_str() {
            "Mint" => {
                if curr_tick >= tick_lower[i] && curr_tick <= tick_upper[i] {
                    last_liq += amount[i];
                }
                active_change.push(0u8);
            }
            "Burn" => {
                if curr_tick >= tick_lower[i] && curr_tick <= tick_upper[i] {
                    last_liq -= amount[i];
                }
                active_change.push(0u8);
            }
            _ => {
                if is_close(last_liq, liquidity[i]) {
                    active_change.push(0u8);
                } else {
                    active_change.push(1u8);
                }
                curr_tick = tick[i];
                last_liq = liquidity[i];
            }
        }
    }

    Ok(active_change)
}

#[pyfunction]
#[pyo3(signature = (block_number, event, sandwich_state, amount0, amount1, verbose=false))]
fn mix_backrun_volumes(
    block_number: Vec<i64>,
    event: Vec<String>,
    sandwich_state: Vec<String>,
    amount0: Vec<f64>,
    amount1: Vec<f64>,
    verbose: bool,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let _ = verbose;
    ensure_same_len5(&block_number, &event, &sandwich_state, &amount0, &amount1)?;
    if block_number.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let mut front_blocks = Vec::new();
    for i in 0..sandwich_state.len() {
        if sandwich_state[i] == "Front_Mix" {
            front_blocks.push(block_number[i]);
        }
    }
    if front_blocks.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let mut useful_idx = Vec::new();
    for i in 0..sandwich_state.len() {
        match sandwich_state[i].as_str() {
            "Front_Mix" | "Mint_Mix" | "Burn_Mix" | "Back_Mix" => useful_idx.push(i),
            _ => {}
        }
    }
    if useful_idx.is_empty() {
        return Err(PyValueError::new_err(
            "Front_Mix rows exist but no mixed labels were found",
        ));
    }

    let useful_blocks: Vec<i64> = useful_idx.iter().map(|&i| block_number[i]).collect();
    let block_slices = build_block_slices(&useful_blocks);
    let mut block_payload: HashMap<i64, (bool, f64, f64)> = HashMap::new();

    for (start, end) in block_slices {
        let block = useful_blocks[start];
        let mut front = None;
        let mut mint = None;
        let mut burn = None;
        let mut back = None;

        for pos in start..end {
            let idx = useful_idx[pos];
            match sandwich_state[idx].as_str() {
                "Front_Mix" if front.is_none() => front = Some(idx),
                "Mint_Mix" if mint.is_none() => mint = Some(idx),
                "Burn_Mix" if burn.is_none() => burn = Some(idx),
                "Back_Mix" if back.is_none() => back = Some(idx),
                _ => {}
            }
        }

        let (f, m, b, k) = match (front, mint, burn, back) {
            (Some(f), Some(m), Some(b), Some(k)) => (f, m, b, k),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Incomplete mixed sequence in block {}",
                    block
                )))
            }
        };

        let payload = if event[f] == "Swap_Y2X" {
            (true, amount0[k], -amount0[f] - amount0[m] + amount0[b])
        } else {
            (false, amount1[k], -amount1[f] - amount1[m] + amount1[b])
        };
        block_payload.insert(block, payload);
    }

    let mut true_x2y = Vec::new();
    let mut expected_x2y = Vec::new();
    let mut true_y2x = Vec::new();
    let mut expected_y2x = Vec::new();

    for block in front_blocks {
        let (is_y2x, true_val, expected) = block_payload
            .get(&block)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("Block {} has Front_Mix but no payload", block)))?;
        if is_y2x {
            expected_y2x.push(expected);
            true_y2x.push(true_val);
        } else {
            expected_x2y.push(expected);
            true_x2y.push(true_val);
        }
    }

    Ok((true_x2y, expected_x2y, true_y2x, expected_y2x))
}

#[pyfunction]
#[pyo3(signature = (series, q=None, alpha=0.05, low_freq_frac=0.10, dfa_min_scale=4, dfa_max_scale=None, dfa_n_scales=20, verbose=false))]
fn long_memory_test(
    series: Vec<f64>,
    q: Option<usize>,
    alpha: f64,
    low_freq_frac: f64,
    dfa_min_scale: usize,
    dfa_max_scale: Option<usize>,
    dfa_n_scales: usize,
    verbose: bool,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, bool)> {
    let _ = verbose;
    if series.len() < 20 {
        return Err(PyValueError::new_err(
            "The time series is too short. Provide at least 20 observations."
        ));
    }
    if !series.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(
            "The time series contains NaN or infinite values."
        ));
    }

    let (lo_two_sided, lo_upper, lo_lower) = lo_modified_rs(&series, q, alpha)?;
    let rs_h = classical_rs_hurst(&series);
    let p_h = periodogram_hurst(&series, low_freq_frac)?;
    let dfa_h = dfa_hurst(&series, dfa_min_scale, dfa_max_scale, dfa_n_scales)?;
    let median_h = median_nan(&[rs_h, p_h, dfa_h]);
    let shows_long_memory = (rs_h > 0.5 && rs_h < 1.0)
        && (p_h > 0.5 && p_h < 1.0)
        && (dfa_h > 0.5 && dfa_h < 1.0);

    Ok((
        lo_two_sided,
        lo_upper,
        lo_lower,
        rs_h,
        p_h,
        dfa_h,
        median_h,
        shows_long_memory,
    ))
}

#[pyfunction]
#[pyo3(signature = (series, block_size=1000, q=None, alpha=0.05, low_freq_frac=0.10, dfa_min_scale=4, dfa_max_scale=None, dfa_n_scales=20))]
fn long_memory_h_dist(
    series: Vec<f64>,
    block_size: usize,
    q: Option<usize>,
    alpha: f64,
    low_freq_frac: f64,
    dfa_min_scale: usize,
    dfa_max_scale: Option<usize>,
    dfa_n_scales: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut h_rs = Vec::<f64>::new();
    let mut h_period = Vec::<f64>::new();
    let mut h_dfa = Vec::<f64>::new();
    let mut h_median = Vec::<f64>::new();

    let mut start = 0usize;
    while start < series.len() {
        let end = (start + block_size).min(series.len());
        let (.., rs_h, p_h, dfa_h, median_h, _) = long_memory_test(
            series[start..end].to_vec(),
            q,
            alpha,
            low_freq_frac,
            dfa_min_scale,
            dfa_max_scale,
            dfa_n_scales,
            false,
        )?;
        h_rs.push(rs_h);
        h_period.push(p_h);
        h_dfa.push(dfa_h);
        h_median.push(median_h);
        start += block_size;
    }

    Ok((h_rs, h_period, h_dfa, h_median))
}

#[pyfunction]
#[pyo3(signature = (price, liquidity, timestamp_ns, mint_tick_lower, mint_tick_upper, mint_amount, mint_block_number, burn_tick_lower, burn_tick_upper, burn_amount, burn_block_number, samp_freq=10, creation_period=false))]
fn provision_summary(
    price: Vec<f64>,
    liquidity: Vec<f64>,
    timestamp_ns: Vec<i64>,
    mint_tick_lower: Vec<f64>,
    mint_tick_upper: Vec<f64>,
    mint_amount: Vec<f64>,
    mint_block_number: Vec<i64>,
    burn_tick_lower: Vec<f64>,
    burn_tick_upper: Vec<f64>,
    burn_amount: Vec<f64>,
    burn_block_number: Vec<i64>,
    samp_freq: i64,
    creation_period: bool,
) -> PyResult<(f64, f64, Vec<f64>, f64, usize, usize, isize, f64)> {
    ensure_same_len3(&price, &liquidity, &timestamp_ns)?;
    ensure_same_len5(
        &mint_tick_lower,
        &mint_tick_upper,
        &mint_amount,
        &mint_block_number,
        &mint_block_number,
    )?;
    ensure_same_len5(
        &burn_tick_lower,
        &burn_tick_upper,
        &burn_amount,
        &burn_block_number,
        &burn_block_number,
    )?;

    let mut price_f = Vec::<f64>::new();
    let mut liq_f = Vec::<f64>::new();
    let mut ts_f = Vec::<i64>::new();
    for i in 0..price.len() {
        if price[i] != 0.0 && !price[i].is_infinite() {
            price_f.push(price[i]);
            liq_f.push(liquidity[i]);
            ts_f.push(timestamp_ns[i]);
        }
    }
    if price_f.is_empty() {
        return Err(PyValueError::new_err("No valid price observations."));
    }

    let n_years = ((ts_f[ts_f.len() - 1] - ts_f[0]) / 86_400_000_000_000i64) as f64 / 365.25;
    let (price_res, liq_res) = resample_last(&ts_f, &price_f, &liq_f, samp_freq);

    let rv_raw_series = diff(&log_vec(&ffill_nan_1d(&price_res)));
    let rv_raw = sumsq_ignore_nan(&rv_raw_series) / n_years;

    let liq_quantiles = quantile_linear(&liq_f, &[0.0, 0.01, 0.05, 0.5]);
    let price_filtered: Vec<f64> = price_res
        .iter()
        .zip(liq_res.iter())
        .map(|(p, l)| if *l > liq_quantiles[2] { *p } else { f64::NAN })
        .collect();
    let rv_series = diff(&log_vec(&ffill_nan_1d(&price_filtered)));
    let rv = sumsq_ignore_nan(&rv_series) / n_years;

    let total_number = mint_tick_lower.len() + burn_tick_lower.len();
    let burn_imbalance = burn_tick_lower.len() as isize - mint_tick_lower.len() as isize;
    let burn_number = burn_tick_lower.len();

    let mut mint_start = 0usize;
    let mut burn_start = 0usize;
    if creation_period {
        if mint_block_number.len() <= 10 {
            return Err(PyValueError::new_err(
                "creation_period=True but there are fewer than 11 mint rows."
            ));
        }
        let first_kept_block = mint_block_number[10];
        while mint_start < mint_block_number.len() && mint_block_number[mint_start] < first_kept_block {
            mint_start += 1;
        }
        while burn_start < burn_block_number.len() && burn_block_number[burn_start] < first_kept_block {
            burn_start += 1;
        }
    }

    let mut tot_liq = 0.0;
    let mut range_weighted_num = 0.0;
    let mut first_mint_block_by_range: HashMap<(u64, u64), i64> = HashMap::new();
    for i in mint_start..mint_tick_lower.len() {
        let tl = mint_tick_lower[i];
        let tu = mint_tick_upper[i];
        let amt = mint_amount[i];
        tot_liq += amt;
        range_weighted_num += (tu - tl).abs() * amt;
        let key = (tl.to_bits(), tu.to_bits());
        first_mint_block_by_range.entry(key).or_insert(mint_block_number[i]);
    }
    let range_weighted = range_weighted_num / tot_liq;

    let mut time_sum = 0.0;
    let mut tot_burnt = 0.0;
    for i in burn_start..burn_tick_lower.len() {
        let key = (burn_tick_lower[i].to_bits(), burn_tick_upper[i].to_bits());
        if let Some(&m_bn) = first_mint_block_by_range.get(&key) {
            let b_bn = burn_block_number[i];
            if m_bn < b_bn {
                let amt = burn_amount[i];
                tot_burnt += amt;
                time_sum += amt * ((b_bn - m_bn) as f64);
            }
        }
    }
    let block_time = time_sum / tot_burnt;

    Ok((
        rv_raw,
        rv,
        liq_quantiles,
        range_weighted,
        total_number,
        burn_number,
        burn_imbalance,
        block_time,
    ))
}

#[pyfunction]
#[pyo3(signature = (sender, amount, cumulative=false))]
fn wallets_activity_sparse(
    sender: Vec<String>,
    amount: Vec<f64>,
    cumulative: bool,
) -> PyResult<(Vec<String>, Vec<Vec<usize>>, Vec<Vec<f64>>)> {
    ensure_same_len2(&sender, &amount)?;

    let mut wallet_keys = Vec::<String>::new();
    let mut wallet_pos = HashMap::<String, usize>::new();
    let mut idx_lists = Vec::<Vec<usize>>::new();
    let mut val_lists = Vec::<Vec<f64>>::new();

    for (i, (wal, amt)) in sender.into_iter().zip(amount.into_iter()).enumerate() {
        let activity = if amt > 0.0 { 1.0 } else { -1.0 };
        let pos = match wallet_pos.entry(wal.clone()) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(v) => {
                let pos = wallet_keys.len();
                wallet_keys.push(wal);
                idx_lists.push(Vec::new());
                val_lists.push(Vec::new());
                v.insert(pos);
                pos
            }
        };
        idx_lists[pos].push(i);
        val_lists[pos].push(activity);
    }

    if cumulative {
        for vals in &mut val_lists {
            let mut csum = 0.0;
            for v in vals {
                csum += *v;
                *v = csum;
            }
        }
    }

    Ok((wallet_keys, idx_lists, val_lists))
}

#[pyfunction]
fn build_wallets_activity_sparse_matrix(
    wallet_keys: Vec<String>,
    idx_lists: Vec<Vec<usize>>,
    val_lists: Vec<Vec<f64>>,
    n_swaps: usize,
) -> PyResult<(Vec<f64>, Vec<usize>, Vec<usize>, usize, usize, Vec<String>)> {
    if wallet_keys.len() != idx_lists.len() || wallet_keys.len() != val_lists.len() {
        return Err(PyValueError::new_err(
            "wallet_keys, idx_lists, and val_lists must have the same length",
        ));
    }

    let mut data = Vec::<f64>::new();
    let mut indices = Vec::<usize>::new();
    let mut indptr = Vec::<usize>::with_capacity(wallet_keys.len() + 1);
    indptr.push(0);

    for row in 0..wallet_keys.len() {
        if idx_lists[row].len() != val_lists[row].len() {
            return Err(PyValueError::new_err(format!(
                "Wallet {}: indices and values length mismatch",
                wallet_keys[row]
            )));
        }
        data.extend_from_slice(&val_lists[row]);
        indices.extend_from_slice(&idx_lists[row]);
        indptr.push(indices.len());
    }

    Ok((data, indices, indptr, wallet_keys.len(), n_swaps, wallet_keys))
}

#[pyfunction]
fn compute_c_split_c_herd_sparse(
    data: Vec<f64>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    n_rows: usize,
    n_cols: usize,
    max_tau: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let _ = n_cols;
    if max_tau < 1 {
        return Err(PyValueError::new_err("max_tau must be >= 1"));
    }
    if indptr.len() != n_rows + 1 {
        return Err(PyValueError::new_err("indptr length must be n_rows + 1"));
    }
    if indices.len() != data.len() {
        return Err(PyValueError::new_err("indices and data length mismatch"));
    }

    let mut y_abs_sum = 0.0;
    for row in 0..n_rows {
        let start = indptr[row];
        let end = indptr[row + 1];
        let mut y = 0.0;
        for &v in &data[start..end] {
            y += v;
        }
        y_abs_sum += y.abs();
    }

    let mut c_split_num = vec![0.0; max_tau];
    let mut c_herd_num = vec![0.0; max_tau];

    for row in 0..n_rows {
        let start = indptr[row];
        let end = indptr[row + 1];
        let row_idx = &indices[start..end];
        let row_val = &data[start..end];

        for i in 0..row_idx.len() {
            let mut j = i + 1;
            while j < row_idx.len() {
                let diff = row_idx[j] - row_idx[i];
                if diff == 0 {
                    j += 1;
                    continue;
                }
                if diff > max_tau {
                    break;
                }
                let prod = row_val[i] * row_val[j];
                if prod >= 0.0 {
                    c_herd_num[diff - 1] += prod;
                } else {
                    c_split_num[diff - 1] += -prod;
                }
                j += 1;
            }
        }
    }

    if y_abs_sum == 0.0 {
        Ok((vec![f64::NAN; max_tau], vec![f64::NAN; max_tau]))
    } else {
        for tau in 0..max_tau {
            c_split_num[tau] /= y_abs_sum;
            c_herd_num[tau] /= y_abs_sum;
        }
        Ok((c_split_num, c_herd_num))
    }
}

#[pyfunction]
fn acf(series: Vec<f64>, lags: usize) -> PyResult<Vec<f64>> {
    if series.is_empty() {
        return Err(PyValueError::new_err("series must be non-empty"));
    }
    if !series.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err("series must contain only finite values"));
    }
    Ok(acf_fft(&series, lags))
}

#[pyfunction]
#[pyo3(signature = (series, n_iter, lags, seed=None, verbose=false))]
fn bootstrap_iid_autocorr(
    py: Python<'_>,
    series: Vec<f64>,
    n_iter: usize,
    lags: usize,
    seed: Option<u64>,
    verbose: bool,
) -> PyResult<Vec<Vec<f64>>> {
    if series.is_empty() {
        return Err(PyValueError::new_err("series must be non-empty"));
    }
    if !series.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err("series must contain only finite values"));
    }

    let n = series.len();
    let base_seed = seed.unwrap_or_else(rand::random::<u64>);

    let out = py.allow_threads(move || {
        (0..n_iter)
            .into_par_iter()
            .map(|iter_idx| {
                if verbose && iter_idx % 100 == 0 {
                    println!("Bootstrap iid iteration {} of {}", iter_idx, n_iter);
                }

                let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(iter_idx as u64));
                let mut resampled = Vec::with_capacity(n);
                for _ in 0..n {
                    resampled.push(series[rng.gen_range(0..n)]);
                }
                acf_fft(&resampled, lags)
            })
            .collect::<Vec<Vec<f64>>>()
    });

    Ok(out)
}

#[pymodule(name = "utils_DevTrad_Rust")]
fn utils_devtrad_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(find_echo, m)?)?;
    m.add_function(wrap_pyfunction!(find_jit, m)?)?;
    m.add_function(wrap_pyfunction!(liq_change, m)?)?;
    m.add_function(wrap_pyfunction!(long_memory_test, m)?)?;
    m.add_function(wrap_pyfunction!(long_memory_h_dist, m)?)?;
    m.add_function(wrap_pyfunction!(mix_backrun_volumes, m)?)?;
    m.add_function(wrap_pyfunction!(provision_summary, m)?)?;
    m.add_function(wrap_pyfunction!(wallets_activity_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(build_wallets_activity_sparse_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(compute_c_split_c_herd_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(acf, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_iid_autocorr, m)?)?;
    Ok(())
}

use std::fmt;
use crate::v2::backend::native::{Buffer, BufferElement, Tensor};
use crate::v2::data::Scalar;

// From ndarray

/// Default threshold, below this element count, we don't ellipsize
const ARRAY_MANY_ELEMENT_LIMIT: usize = 500;
/// Limit of element count for non-last axes before overflowing with an ellipsis.
const AXIS_LIMIT_STACKED: usize = 6;
/// Limit for next to last axis (printed as column)
/// An odd number because one element uses the same space as the ellipsis.
const AXIS_LIMIT_COL: usize = 11;
/// Limit for last axis (printed as row)
/// An odd number because one element uses approximately the space of the ellipsis.
const AXIS_LIMIT_ROW: usize = 11;

#[cfg(test)]
// Test value to use for size of overflowing 2D arrays
const AXIS_2D_OVERFLOW_LIMIT: usize = 22;

/// The string used as an ellipsis.
const ELLIPSIS: &str = "...";

#[derive(Clone, Debug)]
struct FormatOptions {
    axis_collapse_limit: usize,
    axis_collapse_limit_next_last: usize,
    axis_collapse_limit_last: usize,
}

impl FormatOptions {
    pub fn default_for_array(nelem: usize, no_limit: bool) -> Self {
        let default = Self {
            axis_collapse_limit: AXIS_LIMIT_STACKED,
            axis_collapse_limit_next_last: AXIS_LIMIT_COL,
            axis_collapse_limit_last: AXIS_LIMIT_ROW,
        };
        default.set_no_limit(no_limit || nelem < ARRAY_MANY_ELEMENT_LIMIT)
    }

    fn set_no_limit(mut self, no_limit: bool) -> Self {
        if no_limit {
            self.axis_collapse_limit = usize::MAX;
            self.axis_collapse_limit_next_last = usize::MAX;
            self.axis_collapse_limit_last = usize::MAX;
            self
        } else {
            self
        }
    }

    /// Axis length collapse limit before ellipsizing, where `axis_rindex` is
    /// the index of the axis from the back.
    pub fn collapse_limit(&self, axis_rindex: usize) -> usize {
        match axis_rindex {
            0 => self.axis_collapse_limit_last,
            1 => self.axis_collapse_limit_next_last,
            _ => self.axis_collapse_limit,
        }
    }
}

/// Formats the contents of a list of items, using an ellipsis to indicate when
/// the `length` of the list is greater than `limit`.
///
/// # Parameters
///
/// * `f`: The formatter.
/// * `length`: The length of the list.
/// * `limit`: The maximum number of items before overflow.
/// * `separator`: Separator to write between items.
/// * `ellipsis`: Ellipsis for indicating overflow.
/// * `fmt_elem`: A function that formats an element in the list, given the
///   formatter and the index of the item in the list.
fn format_with_overflow(
    f: &mut fmt::Formatter<'_>,
    length: usize,
    limit: usize,
    separator: &str,
    ellipsis: &str,
    fmt_elem: &mut dyn FnMut(&mut fmt::Formatter, usize) -> fmt::Result,
) -> fmt::Result {
    if length == 0 {
        // no-op
    } else if length <= limit {
        fmt_elem(f, 0)?;
        for i in 1..length {
            f.write_str(separator)?;
            fmt_elem(f, i)?
        }
    } else {
        let edge = limit / 2;
        fmt_elem(f, 0)?;
        for i in 1..edge {
            f.write_str(separator)?;
            fmt_elem(f, i)?;
        }
        f.write_str(separator)?;
        f.write_str(ellipsis)?;
        for i in length - edge..length {
            f.write_str(separator)?;
            fmt_elem(f, i)?
        }
    }
    Ok(())
}

fn format<T>(tensor: &Tensor, f: &mut fmt::Formatter<'_>, fmt_opt: &FormatOptions) -> fmt::Result
    where
        T: BufferElement,
{
    format_inner::<T>(tensor, f, fmt_opt, 0, tensor.rank())
}

fn format_inner<T>(
    tensor: &Tensor,
    f: &mut fmt::Formatter<'_>,
    fmt_opt: &FormatOptions,
    depth: usize,
    full_ndim: usize,
) -> fmt::Result
    where
        T: BufferElement,
{
    match tensor.extents() {
        [] => {
            let data = tensor.buffer().as_slice::<T>();
            write!(f, "{}", &data[0])?;
        }

        &[len] => {
            let data = tensor.buffer().as_slice::<T>();
            f.write_str("[")?;
            format_with_overflow(
                f,
                len,
                fmt_opt.collapse_limit(0),
                ", ",
                ELLIPSIS,
                &mut |f, index| write!(f, "{}", &data[tensor.shape().translate_default(index)]),
            )?;
            f.write_str("]")?;
        }
        shape => {
            let blank_lines = "\n".repeat(shape.len() - 2);
            let indent = " ".repeat(depth + 1);
            let separator = format!(",\n{}{}", blank_lines, indent);

            f.write_str("[")?;
            let limit = fmt_opt.collapse_limit(full_ndim - depth - 1);
            format_with_overflow(f, shape[0], limit, &separator, ELLIPSIS, &mut |f, index| {
                format_inner::<T>(&tensor.index(index, 0), f, fmt_opt, depth + 1, full_ndim)
            })?;
            f.write_str("]")?;
        }
    }
    Ok(())
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_opt = FormatOptions::default_for_array(self.size(), f.alternate());

        match self.buffer() {
            Buffer::Float(_) => format::<f32>(self, f, &fmt_opt),
            Buffer::Int(_) => format::<i32>(self, f, &fmt_opt),
        }
    }
}

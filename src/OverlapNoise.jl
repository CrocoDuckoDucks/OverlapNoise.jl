module OverlapNoise

import FFTW
import Statistics
import WAV

"""
    calcOverlapSamples(wLength, pOverlap)

Given a window length wLength and a proportion of overlap pOverlap, calculate
the length of the overalp section between two windows, in samples.
"""
function calcOverlapSamples(
    wLength::Integer,
    pOverlap::Real
    )

    return floor(Integer, pOverlap * wLength)

end

"""
    overlapWindowLimits(wLength, pOverlap, nWindow)

Given a window length wLength and a proportion of overlap pOverlap, return the
lower and upper samples of window number nWindow.
"""
function overlapWindowLimits(
    wLength::Integer,
    pOverlap::Real,
    nWindow::Integer
    )

    oLength = calcOverlapSamples(wLength, pOverlap)

    return (nWindow - 1) * (wLength - oLength) + 1,
        wLength + (nWindow - 1) * (wLength - oLength)

end

"""
    overlapNoise(wLength, pOverlap, nWindows, level)

Given a window length wLength, a proportion of overlap pOverlap and a total
number of windows nWindows, return a chunk of white uniform noise covering all
the requested windows. Optionally specify the noise level in dBFS.
"""
function overlapNoise(
    wLength::Integer,
    pOverlap::Real,
    nWindows::Integer,
    level::Real = 0.0
    )

    _, nLength = overlapWindowLimits(wLength, pOverlap, nWindows)

    A = exp10(level / 20.0)

    return 2A * rand(nLength) .- A

end

"""
    segmentSignal(x, wLength, pOverlap, nWindows)

Given a window length wLength, a proportion of overlap pOverlap and a total
number of windows nWindows, segment a signal x into the various overlapping
windows and return them into a matrix wLength by nWindows size.
"""
function segmentSignal(
    x::AbstractVector{<:Real},
    wLength::Integer,
    pOverlap::Real,
    nWindows::Integer,
    )

    X = zeros(eltype(x), wLength, nWindows)

    for n in 1:nWindows

        l, u = overlapWindowLimits(wLength, pOverlap, n)

        if u > length(x)
            break
        end

        X[:, n] = x[l:u]

    end

    return X

end

"""
    nyquistSample(wLength)

For a given window length wLength, calculate the corresponding Nyquist sample.
"""
function nyquistSample(
    wLength::Integer
    )

    return div(wLength, 2) + 1

end

"""
    rfftFrequency(wLength, Fs)

Given the window length wLength and the sample rate Fs, calculate the frequency
axis for the FFT of a real valued signal contained in a window (operated by
using FFTW rfft())
"""
function rfftFrequency(
    wLength::Integer,
    Fs::Real
    )

    nyq = nyquistSample(wLength)

    return (0:(nyq - 1)) * Fs / 2nyq

end

"""
    rfftFrequencyResolution(wLength, Fs)

Given the window length wLength and the sample rate Fs, calculate the frequency
resolution of the frequency axis for the FFT of a real valued signal contained
in a window (operated by using FFTW rfft()).

See also [`rfft`](@ref).
"""
function rfftFrequencyResolution(
    wLength::Integer,
    Fs::Real
    )

    return Fs / (2 * nyquistSample(wLength))

end

"""
    genericRMS(x)

Calculate the RMS value of a signal x
"""
function genericRMS(
    x::AbstractArray
    )

return sqrt(mean(abs2.(x[:])))

end

"""
    applyWindow(X, window)

Apply a smoothing window contained in window to a segmented signal contained in
the matrix X, produced with segmentSignal. The results are scaled to preserve
the L2 norm of the segments.

See also [`segmentSignal`](@ref).
"""
function applyWindow(
    X::AbstractMatrix{<:Real},
    window::AbstractVector{<:Real}
    )

    Xw = X .* window

    A = sum(abs2, X, dims=1) ./ sum(abs2, Xw, dims=1)

    return sqrt.(A) .* Xw

end

"""
    applyrfft(X)

Apply rfft to a segmented signal contained in the matrix X, produced with
segmentSignal. Scale the results to preserve the L2 norm.

See also [`rfft`](@ref).
See also [`segmentSignal`](@ref).
"""
function applyrfft(
    X::AbstractMatrix{<:Real}
    )

    return FFTW.rfft(X, 1) * sqrt(2.0 / size(X, 1))

end

"""
    lin2dB(x)

Calculate the dBFS values of the linear units values contained in array x.
"""
function lin2dB(x::AbstractArray)
    return 20.0 * log10.(abs.(x))
end

"""
    lin2rad(X)

Calculate the angle of the values contained in array x.
"""
function lin2rad(x::AbstractArray)
    return angle.(x)
end

"""
    saveAudio(x, Fs, fName)

Save contents of array x as a wav file with sample rate Fs and file name fName.
"""
function saveAudio(x::AbstractArray{<:Real}, Fs::Real, fName::String)
    WAV.wavwrite(x, fName, Fs=Fs)
end

"""
    loadAudio(fName)

Load samples and sample rate of the wav file named fName.
"""
function loadAudio(fName::String)

    x, Fs, _, _ = WAV.wavread(fName)

    return x, Float64(Fs)

end

"""
    transferCalc(x, y, wLength, pOverlap, nWindows, window)

Given a system input broadband noise x and a recorded system output y, calculate
the system frequency response by frequency domain analysis of x and y using
nWindows windows wLength samples long, with pOverlap proportion of overlap. The
window samples are contained in window. x can be gereanted with overlapNoise.

See also [`overlapNoise`](@ref).
"""
function transferCalc(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    wLength::Integer,
    pOverlap::Real,
    nWindows::Integer,
    window::AbstractVector{<:Real}
    )

    X = applyrfft(
        applyWindow(segmentSignal(x, wLength, pOverlap, nWindows), window)
        )

    Y = applyrfft(
        applyWindow(segmentSignal(y, wLength, pOverlap, nWindows), window)
        )

    return Statistics.mean(Y .* conj(X), dims=2) ./
        Statistics.mean(abs2.(X), dims=2)

end

end # module

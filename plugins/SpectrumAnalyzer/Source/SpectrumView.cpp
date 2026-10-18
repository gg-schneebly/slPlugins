#include "SpectrumView.h"

namespace
{
    /** Formats a frequency as a short axis label, e.g. 50, 500, 2k, 2.5k. */
    juce::String formatFrequency (float frequency)
    {
        if (frequency >= 1000.0f)
        {
            auto khz = frequency / 1000.0f;
            return (std::abs (khz - std::round (khz)) < 0.01f ? juce::String (juce::roundToInt (khz))
                                                               : juce::String (khz, 1)) + "k";
        }

        return juce::String (juce::roundToInt (frequency));
    }

    /** Picks a 1-2-5 step that splits range into roughly targetDivisions. */
    float niceStep (float range, int targetDivisions)
    {
        auto rough = range / (float) targetDivisions;
        auto magnitude = std::pow (10.0f, std::floor (std::log10 (rough)));

        for (auto multiple : { 1.0f, 2.0f, 5.0f })
            if (multiple * magnitude >= rough)
                return multiple * magnitude;

        return 10.0f * magnitude;
    }
}

//==============================================================================
SpectrumView::SpectrumView()
{
    setColour (traceColourId, juce::Colours::white);
    setColour (fillColourId,  juce::Colours::white.withAlpha (0.08f));
    setColour (gridColourId,  juce::Colours::grey.withAlpha (0.35f));
    setColour (textColourId,  juce::Colours::grey);

    setInterceptsMouseClicks (false, false);
    resetAnalysis();
    startTimerHz (refreshHz);
}

SpectrumView::~SpectrumView()
{
    stopTimer();
}

void SpectrumView::pushSamples (const float* samples, int numSamples)
{
    // Drops samples if the message thread falls behind; it only ever wants the latest audio
    const auto scope = fifo.write (juce::jmin (numSamples, fifo.getFreeSpace()));

    if (scope.blockSize1 > 0)
        std::copy (samples, samples + scope.blockSize1, fifoBuffer.begin() + scope.startIndex1);

    if (scope.blockSize2 > 0)
        std::copy (samples + scope.blockSize1, samples + scope.blockSize1 + scope.blockSize2,
                   fifoBuffer.begin() + scope.startIndex2);
}

void SpectrumView::setFftOrder (int newOrder)
{
    newOrder = juce::jlimit (10, 15, newOrder);

    if (newOrder != fftOrder)
    {
        fftOrder = newOrder;
        resetAnalysis();
        repaint();
    }
}

void SpectrumView::setLogFrequencyDisplay (bool shouldDisplayLog)
{
    logFrequency = shouldDisplayLog;
    repaint();
}

void SpectrumView::setSampleRate (double newSampleRate)
{
    // No repaint here: this may be called from the audio thread, the next timer tick picks it up
    if (newSampleRate > 0.0)
        sampleRate = newSampleRate;
}

void SpectrumView::setDrawGrid (bool shouldDraw)
{
    shouldDrawGrid = shouldDraw;
    repaint();
}

//==============================================================================
void SpectrumView::resetAnalysis()
{
    const auto fftSize = 1 << fftOrder;

    fft = std::make_unique<juce::dsp::FFT> (fftOrder);
    // normalise = true gives the window unity gain, so a full-scale sine reads 0 dB after scaling by 2 / N
    window = std::make_unique<juce::dsp::WindowingFunction<float>> ((size_t) fftSize,
                                                                    juce::dsp::WindowingFunction<float>::hann, true);

    history.assign ((size_t) fftSize, 0.0f);
    fftData.assign ((size_t) fftSize * 2, 0.0f);
    levelsDb.assign ((size_t) fftSize / 2 + 1, minDecibels);
    samplesSinceLastFrame = 0;
}

void SpectrumView::timerCallback()
{
    if (processAvailableSamples())
        repaint();
}

bool SpectrumView::processAvailableSamples()
{
    const auto fftSize = (int) history.size();

    // After a stall (e.g. hidden window) skip stale audio so the display stays current
    const auto maxBacklog = fftSize + hopSize * 4;
    if (fifo.getNumReady() > maxBacklog)
        fifo.read (fifo.getNumReady() - maxBacklog);

    bool changed = false;

    while (fifo.getNumReady() > 0)
    {
        const auto chunk = juce::jmin (fifo.getNumReady(), hopSize - samplesSinceLastFrame);
        const auto scope = fifo.read (chunk);

        std::move (history.begin() + chunk, history.end(), history.begin());
        auto dest = history.end() - chunk;
        dest = std::copy (fifoBuffer.begin() + scope.startIndex1,
                          fifoBuffer.begin() + scope.startIndex1 + scope.blockSize1, dest);
        std::copy (fifoBuffer.begin() + scope.startIndex2,
                   fifoBuffer.begin() + scope.startIndex2 + scope.blockSize2, dest);

        samplesSinceLastFrame += chunk;

        if (samplesSinceLastFrame >= hopSize)
        {
            samplesSinceLastFrame = 0;
            computeSpectrumFrame();
            changed = true;
        }
    }

    return changed;
}

void SpectrumView::computeSpectrumFrame()
{
    const auto fftSize = (int) history.size();

    std::copy (history.begin(), history.end(), fftData.begin());
    std::fill (fftData.begin() + fftSize, fftData.end(), 0.0f);

    window->multiplyWithWindowingTable (fftData.data(), (size_t) fftSize);
    fft->performFrequencyOnlyForwardTransform (fftData.data(), true);

    const auto scale = 2.0f / (float) fftSize;
    const auto frameSeconds = (float) hopSize / (float) sampleRate;
    const auto release = std::exp (-frameSeconds * 1000.0f / releaseMs);

    for (size_t bin = 0; bin < levelsDb.size(); ++bin)
    {
        const auto db = juce::Decibels::gainToDecibels (fftData[bin] * scale, minDecibels);
        auto& level = levelsDb[bin];

        // Instant attack, short release: rises and falls both show up within a few frames
        level = db > level ? db : db + (level - db) * release;
    }
}

//==============================================================================
float SpectrumView::frequencyToX (float frequency, float width) const
{
    const auto nyquist = (float) sampleRate * 0.5f;

    if (logFrequency)
        return std::log (frequency / minLogFrequency) / std::log (nyquist / minLogFrequency) * width;

    return frequency / nyquist * width;
}

float SpectrumView::xToFrequency (float x, float width) const
{
    const auto nyquist = (float) sampleRate * 0.5f;
    const auto proportion = x / width;

    if (logFrequency)
        return minLogFrequency * std::pow (nyquist / minLogFrequency, proportion);

    return proportion * nyquist;
}

float SpectrumView::levelAtFrequencyRange (float lowFrequency, float highFrequency) const
{
    const auto binWidth = (float) sampleRate / (float) history.size();
    const auto lastBin = (float) levelsDb.size() - 1.0f;
    const auto lowBin  = juce::jlimit (0.0f, lastBin, lowFrequency / binWidth);
    const auto highBin = juce::jlimit (0.0f, lastBin, highFrequency / binWidth);

    // Pixel narrower than a bin (low end of the log axis): interpolate between neighbouring bins
    if (highBin - lowBin < 1.0f)
    {
        const auto centre = (lowBin + highBin) * 0.5f;
        const auto below = (size_t) centre;
        const auto above = juce::jmin (below + 1, levelsDb.size() - 1);
        const auto t = centre - (float) below;

        return levelsDb[below] + (levelsDb[above] - levelsDb[below]) * t;
    }

    // Pixel spans several bins: show the loudest so narrow peaks don't vanish
    auto level = minDecibels;

    for (auto bin = (size_t) std::ceil (lowBin); bin <= (size_t) highBin; ++bin)
        level = juce::jmax (level, levelsDb[bin]);

    return level;
}

float SpectrumView::decibelsToY (float decibels, float height) const
{
    const auto proportion = (juce::jlimit (minDecibels, maxDecibels, decibels) - minDecibels) / (maxDecibels - minDecibels);
    return height - proportion * height;
}

//==============================================================================
void SpectrumView::paint (juce::Graphics& g)
{
    const auto area = getLocalBounds().toFloat();

    if (area.isEmpty())
        return;

    if (shouldDrawGrid)
        drawGrid (g, area);

    drawSpectrum (g, area);
}

void SpectrumView::drawGrid (juce::Graphics& g, juce::Rectangle<float> area)
{
    const auto nyquist = (float) sampleRate * 0.5f;
    const auto labelHeight = 14.0f;

    g.setFont (juce::FontOptions (11.0f));

    auto drawFrequencyLine = [&] (float frequency, bool withLabel)
    {
        const auto x = area.getX() + frequencyToX (frequency, area.getWidth());

        g.setColour (findColour (gridColourId));
        g.drawVerticalLine (juce::roundToInt (x), area.getY(), area.getBottom());

        if (withLabel)
        {
            g.setColour (findColour (textColourId));
            g.drawText (formatFrequency (frequency),
                        juce::Rectangle<float> (x + 3.0f, area.getBottom() - labelHeight, 40.0f, labelHeight),
                        juce::Justification::centredLeft, false);
        }
    };

    if (logFrequency)
    {
        for (auto decade = minLogFrequency; decade < nyquist; decade *= 10.0f)
        {
            for (int multiple = 1; multiple < 10; ++multiple)
            {
                const auto frequency = decade * (float) multiple;

                if (frequency > minLogFrequency && frequency < nyquist)
                    drawFrequencyLine (frequency, multiple == 1 || multiple == 2 || multiple == 5);
            }
        }
    }
    else
    {
        const auto step = niceStep (nyquist, juce::jmax (2, juce::roundToInt (area.getWidth() / 90.0f)));

        for (auto frequency = step; frequency < nyquist; frequency += step)
            drawFrequencyLine (frequency, true);
    }

    for (auto db = maxDecibels - 20.0f; db > minDecibels; db -= 20.0f)
    {
        const auto y = area.getY() + decibelsToY (db, area.getHeight());

        g.setColour (findColour (gridColourId));
        g.drawHorizontalLine (juce::roundToInt (y), area.getX(), area.getRight());

        g.setColour (findColour (textColourId));
        g.drawText (juce::String (juce::roundToInt (db)) + " dB",
                    juce::Rectangle<float> (area.getX() + 3.0f, y - labelHeight, 50.0f, labelHeight),
                    juce::Justification::bottomLeft, false);
    }

    g.setColour (findColour (gridColourId));
    g.drawRect (area);
}

void SpectrumView::drawSpectrum (juce::Graphics& g, juce::Rectangle<float> area)
{
    juce::Path trace;
    const auto width = area.getWidth();
    const auto columns = juce::roundToInt (width);

    for (int column = 0; column <= columns; ++column)
    {
        const auto x = (float) column;
        const auto level = levelAtFrequencyRange (xToFrequency (x, width), xToFrequency (x + 1.0f, width));
        const auto point = juce::Point<float> (area.getX() + x, area.getY() + decibelsToY (level, area.getHeight()));

        if (column == 0)
            trace.startNewSubPath (point);
        else
            trace.lineTo (point);
    }

    auto fill = trace;
    fill.lineTo (area.getRight(), area.getBottom());
    fill.lineTo (area.getX(), area.getBottom());
    fill.closeSubPath();

    g.setColour (findColour (fillColourId));
    g.fillPath (fill);

    g.setColour (findColour (traceColourId));
    g.strokePath (trace, juce::PathStrokeType (1.5f));
}

#pragma once

#include <JuceHeader.h>

//==============================================================================
/** Real-time magnitude spectrum of one audio channel.

    Samples are pushed from the audio thread through a lock-free fifo; the
    message thread runs overlapping Hann-windowed FFTs and draws the result on
    either a true logarithmic axis (minLogFrequency .. Nyquist) or a linear
    axis (0 Hz .. Nyquist), with labelled frequency and level grid lines.

    Unlike a peak-hold display, levels fall as fast as they rise (only a short
    release smooths flicker), so cuts such as a high-pass sweep show up without
    lag.
*/
class SpectrumView : public juce::Component,
                     private juce::Timer
{
public:
    enum ColourIds
    {
        traceColourId = 0x2a01000,
        fillColourId  = 0x2a01001,
        gridColourId  = 0x2a01002,
        textColourId  = 0x2a01003,
    };

    SpectrumView();
    ~SpectrumView() override;

    /** Adds samples to the analysis fifo. Safe to call from the audio thread.
        @param samples     mono sample data
        @param numSamples  number of samples in samples
    */
    void pushSamples (const float* samples, int numSamples);

    /** Sets the FFT size as a power of two (e.g. 13 = 8192 points). Larger
        sizes resolve low frequencies better but react more slowly.
        @param newOrder  log2 of the FFT size, clamped to 10..15
    */
    void setFftOrder (int newOrder);

    /** Switches between a logarithmic and a linear frequency axis. */
    void setLogFrequencyDisplay (bool shouldDisplayLog);

    /** Sets the sample rate used for the frequency axis. Safe to call from any thread.
        @param newSampleRate  in Hz; values <= 0 are ignored
    */
    void setSampleRate (double newSampleRate);

    /** Enables drawing of the grid and labels. When several views are stacked,
        enable it on one of them only.
    */
    void setDrawGrid (bool shouldDrawGrid);

    //==============================================================================
    void paint (juce::Graphics&) override;

    static constexpr float minLogFrequency = 10.0f;
    static constexpr float minDecibels     = -100.0f;
    static constexpr float maxDecibels     = 0.0f;

private:
    void timerCallback() override;

    /** Pulls samples from the fifo and runs one FFT per hop.
        @returns true if the spectrum changed
    */
    bool processAvailableSamples();
    void computeSpectrumFrame();
    void resetAnalysis();

    float frequencyToX (float frequency, float width) const;
    float xToFrequency (float x, float width) const;
    float levelAtFrequencyRange (float lowFrequency, float highFrequency) const;
    float decibelsToY (float decibels, float height) const;

    void drawGrid (juce::Graphics&, juce::Rectangle<float> area);
    void drawSpectrum (juce::Graphics&, juce::Rectangle<float> area);

    //==============================================================================
    static constexpr int fifoSize   = 1 << 16;
    static constexpr int hopSize    = 512;
    static constexpr float releaseMs = 50.0f;
    static constexpr int refreshHz  = 60;

    juce::AbstractFifo fifo { fifoSize };
    std::vector<float> fifoBuffer = std::vector<float> ((size_t) fifoSize);

    int fftOrder = 13;
    std::unique_ptr<juce::dsp::FFT> fft;
    std::unique_ptr<juce::dsp::WindowingFunction<float>> window;

    std::vector<float> history;         // last fftSize samples, oldest first
    std::vector<float> fftData;         // 2 * fftSize work buffer
    std::vector<float> levelsDb;        // smoothed level per bin, in dB
    int samplesSinceLastFrame = 0;

    std::atomic<double> sampleRate { 48000.0 };
    bool logFrequency = true;
    bool shouldDrawGrid = true;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpectrumView)
};

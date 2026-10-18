#pragma once

#include "../JuceLibraryCode/JuceHeader.h"

class PluginEditor;

#define PARAM_MODE     "mode"
#define PARAM_LOG      "log"
#define PARAM_RES      "res"

//==============================================================================
/**
*/
class PluginProcessor : public gin::Processor
{
public:
    //==============================================================================
    PluginProcessor();
    ~PluginProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    void processBlock (juce::AudioSampleBuffer&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================

    /** Maps the Resolution parameter (0, 1, 2) to an FFT size (4096, 8192, 16384). */
    static int resolutionToFftOrder (int resolution)  { return 12 + juce::jlimit (0, 2, resolution); }
    static int resolutionToFftSize (int resolution)   { return 1 << resolutionToFftOrder (resolution); }

    juce::CriticalSection lock;
    PluginEditor* editor = nullptr;

private:    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginProcessor)
};

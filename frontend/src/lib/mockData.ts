/**
 * Demo data — used ONLY when VITE_USE_MOCK_DATA=true, so the UI can be reviewed
 * without a running backend. Never merged into a real API response: the screens
 * read either the API or this module, never both.
 *
 * The shapes are the real contract shapes from lib/api.ts, and the wording is
 * the approved mockup copy, so switching to live data changes nothing visually.
 */

import type { AnalyzeResponse } from './api'
import type { AnalysisRecord } from './analysisStore'

export const MOCK_DISCLAIMER =
  'AI-assisted analysis. SkinAIBot is a clinical decision-support prototype, not a diagnosis. ' +
  'Please consult a qualified dermatologist about anything that concerns you.'

export const mockAnalysis: AnalyzeResponse = {
  analysis_id: 'A-4192',
  status: 'completed',
  model_version: 'demo-efficientnet-b3',
  predictions: [
    { label: 'seborrheic_keratosis', confidence: 0.71 },
    { label: 'melanocytic_nevus', confidence: 0.18 },
    { label: 'basal_cell_carcinoma', confidence: 0.06 },
    { label: 'actinic_keratosis', confidence: 0.03 },
  ],
  confidence_status: 'moderate',
  explanation:
    'The border of the growth is even and its surface texture is rough and slightly raised. ' +
    'Colour is uniform, and the model found none of the irregular pigment patterns it associates ' +
    'with the highest-risk classes.',
  recommendation:
    'Book a routine dermatology appointment within four weeks. Photograph the same spot every ' +
    'fortnight and go sooner if it bleeds, itches persistently, or changes shape.',
  disclaimer: MOCK_DISCLAIMER,
}

function record(
  id: string,
  createdAt: string,
  site: string,
  filename: string,
  result: AnalyzeResponse | null,
  error: string | null = null,
): AnalysisRecord {
  return {
    analysis_id: id,
    created_at: createdAt,
    body_site: site,
    upload: {
      id: `u-${id}`,
      original_filename: filename,
      content_type: 'image/jpeg',
      size_bytes: 2_841_077,
    },
    image_url: null,
    result,
    error,
  }
}

export const mockAnalysisHistory: AnalysisRecord[] = [
  record('A-4192', '2026-08-11T03:48:00Z', 'Left forearm', 'left-forearm-01.jpg', mockAnalysis),
  record('A-3980', '2026-07-28T09:12:00Z', 'Left forearm', 'left-forearm-00.jpg', {
    ...mockAnalysis,
    analysis_id: 'A-3980',
    confidence_status: 'high',
    predictions: [
      { label: 'seborrheic_keratosis', confidence: 0.83 },
      { label: 'melanocytic_nevus', confidence: 0.11 },
      { label: 'actinic_keratosis', confidence: 0.06 },
    ],
  }),
  record('A-3611', '2026-06-02T17:40:00Z', 'Upper back', 'upper-back-01.jpg', {
    ...mockAnalysis,
    analysis_id: 'A-3611',
    confidence_status: 'high',
    predictions: [
      { label: 'melanocytic_nevus', confidence: 0.91 },
      { label: 'seborrheic_keratosis', confidence: 0.07 },
      { label: 'dermatofibroma', confidence: 0.02 },
    ],
  }),
  record('A-3104', '2026-03-14T11:05:00Z', 'Right shin', 'right-shin-01.jpg', {
    ...mockAnalysis,
    analysis_id: 'A-3104',
    confidence_status: 'low',
    predictions: [
      { label: 'dermatofibroma', confidence: 0.44 },
      { label: 'melanocytic_nevus', confidence: 0.31 },
      { label: 'seborrheic_keratosis', confidence: 0.25 },
    ],
  }),
  record(
    'A-3098',
    '2026-03-02T08:20:00Z',
    'Right shin',
    'right-shin-00.jpg',
    null,
    'The photograph was too blurred to analyse.',
  ),
]

export const mockChatReplies: string[] = [
  '0.71 means the model considers that class the best match but is not certain. It sits in the ' +
    'middle band — high enough to be the likely answer, low enough that the alternatives stay on ' +
    'the table. A clinician should look at the spot directly.',
  'Four changes are worth acting on early: bleeding without being knocked, a border that becomes ' +
    'uneven, new colours inside the spot, or steady growth. Photograph it fortnightly from the ' +
    'same distance so change is visible rather than remembered.',
]

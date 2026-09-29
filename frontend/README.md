# GenreLab Frontend

A visually polished React + Vite frontend for music genre classification using three deep learning models (CNN, LSTM, ResNet18).

## Features

- **Modern Dark UI**: Cinema-inspired interface with glass effects and smooth animations
- **Real-time Audio Analysis**: Upload audio and get instant genre predictions
- **Multi-Model Support**: Compare predictions from CNN, LSTM, or ResNet18
- **Audio Visualization**: Built-in player with waveform progress display
- **Responsive Design**: Optimized for desktop, tablet, and mobile
- **Live API Integration**: Direct integration with FastAPI backend

## Quick Start

### Prerequisites

- Node.js 16+ and npm
- FastAPI backend running at `http://127.0.0.1:8000`

### Installation

```bash
cd frontend
npm install
```

### Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` if your backend is running on a different URL:

```
VITE_API_URL=http://127.0.0.1:8000
```

### Development

Start the development server:

```bash
npm run dev
```

The application will open at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## Usage

1. **Upload Audio**: Drag and drop an audio file or click to browse
   - Supported formats: MP3, WAV, FLAC, and other common audio formats
   - Audio will be processed as 30-second mono at 22.05 kHz (matching training pipeline)

2. **Select Model**:
   - **CNN**: Convolutional Neural Network (102,416 parameters)
   - **LSTM**: Long Short-Term Memory (273,488 parameters)
   - **ResNet18**: Residual Network (11,178,448 parameters)
   - **Compare All**: Run all three models simultaneously

3. **Analyze**: Click "Analyze Track" to run inference

4. **View Results**:
   - Single model: Predicted genre, confidence %, probability distribution
   - Compare: Side-by-side predictions with consensus genre

## Architecture

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── Header.tsx
│   │   ├── AudioUploader.tsx
│   │   ├── AudioPlayer.tsx
│   │   ├── ModelSelector.tsx
│   │   ├── AnalysisButton.tsx
│   │   ├── PredictionResult.tsx
│   │   ├── ModelComparison.tsx
│   │   ├── ModelInfo.tsx
│   │   └── TechnicalDetails.tsx
│   ├── services/
│   │   └── api.ts           # API client
│   ├── types/
│   │   └── api.ts           # TypeScript types
│   ├── App.tsx              # Main application
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles & animations
├── index.html               # HTML template
├── vite.config.ts           # Vite configuration
├── tailwind.config.js       # Tailwind CSS config
├── postcss.config.js        # PostCSS config
└── tsconfig.json            # TypeScript config
```

## Technologies

- **React 18**: UI framework
- **Vite**: Fast build tool
- **TypeScript**: Type safety
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library
- **HTML5 Audio API**: Audio playback and visualization

## Design Principles

- **Dark aesthetic**: Deep slate and violet with cyan accents
- **Glass morphism**: Subtle backdrop blur effects
- **Smooth animations**: Tasteful transitions and progressive reveal
- **Responsive**: Mobile-first design approach
- **Accessible**: Keyboard navigation and ARIA labels

## Model Information

### CNN (Convolutional Neural Network)
- Analyzes spatial patterns in mel-spectrograms
- 3 convolutional blocks with BatchNorm and pooling
- 102,416 parameters
- Best for: Pattern recognition in frequency domain

### LSTM (Long Short-Term Memory)
- Models temporal sequence information
- 2-layer LSTM with classifier head
- 273,488 parameters
- Best for: Temporal dependencies across time

### ResNet18 (Residual Network)
- Deep spectrogram feature extraction
- Modified for 1-channel input
- 11,178,448 parameters
- Best for: Complex hierarchical feature learning

## Audio Processing

All models use identical preprocessing:
1. Load audio at 22.05 kHz (mono)
2. Enforce 30-second duration (pad/truncate)
3. Extract 128-bin mel spectrogram (FFT 2048, hop 512)
4. Convert to dB scale: range [-80, 0] dB
5. Normalize: (mel + 80) / 80 → [0, 1]
6. Model-specific adjustments (channel dims, temporal downsampling for LSTM)

## API Endpoints

- `GET /health` - API health check
- `POST /predict` - Single model prediction
- `POST /compare` - Multi-model comparison

## Troubleshooting

**"API Offline" error:**
- Ensure FastAPI backend is running at the configured URL
- Check `VITE_API_URL` environment variable

**Audio file not accepted:**
- Use MP3, WAV, or FLAC format
- Ensure file is not corrupted

**Slow analysis:**
- First inference may be slower due to model loading
- Subsequent analyses should be faster

**Responsive layout issues:**
- Hard refresh browser (Ctrl+Shift+R or Cmd+Shift+R)
- Check browser dev tools responsive design mode

## Performance

- **FP32 ResNet18 (GPU)**: 3.66 ms/sample, 63.69% test accuracy
- **CNN (GPU)**: ~58% test accuracy
- **LSTM (GPU)**: 55.87% test accuracy

## Browser Support

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

College Capstone Project • 2024

## Authors

Music Genre Classification Team

# Robot Production Monitor - Modern Web UI

A modern, responsive React TypeScript application for monitoring and controlling industrial robot production systems.

## Features

### 🚀 **Real-time Monitoring**
- Live dashboard with real-time production metrics
- WebSocket connectivity for instant updates
- Robot status monitoring and control
- Production alerts and notifications

### 📊 **Advanced Analytics**
- Interactive charts and visualizations
- KPI tracking and trend analysis
- Anomaly detection and alerts
- Production performance insights

### 🤖 **Robot Management**
- Multi-robot status overview
- Individual robot control panels
- Parameter optimization recommendations
- Performance analytics

### 📈 **Production Optimization**
- AI/ML-powered optimization suggestions
- Quality prediction and analysis
- Maintenance scheduling insights
- Energy efficiency monitoring

### 📱 **Responsive Design**
- Mobile-friendly interface
- Progressive Web App (PWA) support
- Dark mode support
- Accessibility compliant

## Technology Stack

- **Frontend**: React 18 with TypeScript
- **UI Framework**: Material-UI (MUI) v5
- **Charts**: Recharts & MUI X Charts
- **State Management**: React Query
- **Routing**: React Router v6
- **Styling**: Emotion (CSS-in-JS)
- **Notifications**: React Hot Toast
- **WebSocket**: Native WebSocket API

## Prerequisites

- Node.js 16+
- npm or yarn
- Access to Robot Production System backend services

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd modern_web_ui
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` file with your configuration:
```
REACT_APP_API_BASE_URL=http://localhost:8080/api
REACT_APP_WS_URL=ws://localhost:8080/ws
REACT_APP_ENVIRONMENT=development
```

### Development

Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

### Production Build

Create a production build:
```bash
npm run build
```

The build files will be created in the `build/` directory.

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Charts/         # Chart components
│   ├── Layout/         # Layout components
│   ├── Metrics/        # Metric display components
│   ├── Robots/         # Robot-related components
│   └── Alerts/         # Alert components
├── pages/              # Page components
│   ├── Dashboard.tsx   # Main dashboard
│   ├── Analytics.tsx   # Analytics page
│   ├── Robots.tsx      # Robot management
│   ├── Production.tsx  # Production monitoring
│   ├── Optimization.tsx # AI optimization
│   └── Settings.tsx    # Settings page
├── hooks/              # Custom React hooks
├── services/           # API services
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
└── styles/             # Global styles
```

## Key Components

### Dashboard
- Real-time production metrics
- Robot status overview
- Active alerts panel
- KPI trends and charts

### Analytics
- Detailed production analytics
- Trend analysis
- Anomaly detection
- Performance insights

### Robot Management
- Individual robot status
- Parameter controls
- Performance metrics
- Maintenance alerts

### Optimization
- AI-powered recommendations
- Quality predictions
- Energy efficiency analysis
- Process optimization

## API Integration

The frontend communicates with the following backend services:

- **Production System API** (`/api/production/*`)
- **Robot Control API** (`/api/robots/*`)
- **Metrics API** (`/api/metrics/*`)
- **Analytics API** (`/api/analytics/*`)
- **Optimization API** (`/api/optimization/*`)
- **WebSocket API** (`/ws/*`)

## Configuration

### Environment Variables

- `REACT_APP_API_BASE_URL`: Base URL for API requests
- `REACT_APP_WS_URL`: WebSocket server URL
- `REACT_APP_ENVIRONMENT`: Application environment

### Theme Customization

The Material-UI theme can be customized in `src/App.tsx`:

```typescript
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    // ... other theme options
  },
});
```

## WebSocket Integration

Real-time updates are handled through WebSocket connections:

```typescript
const { lastMessage, connectionStatus } = useWebSocket('ws://localhost:8080/ws/dashboard');
```

Message types:
- `system_alert`: System-wide alerts
- `production_update`: Production status changes
- `quality_alert`: Quality-related alerts
- `robot_status_update`: Robot status changes

## Testing

Run the test suite:
```bash
npm test
```

Run tests with coverage:
```bash
npm test -- --coverage
```

## Build and Deployment

### Production Build
```bash
npm run build
```

### Static Analysis
```bash
npm run type-check
```

### Docker Deployment
```bash
# Build Docker image
docker build -t robot-production-monitor-ui .

# Run container
docker run -p 3000:80 robot-production-monitor-ui
```

## Features by Component

### MetricCard
- Displays key performance indicators
- Shows trends and targets
- Configurable colors and icons
- Loading states

### AlertList
- Real-time alert notifications
- Alert acknowledgment
- Filtering and sorting
- Priority-based display

### KPITrendChart
- Interactive trend visualization
- Multiple KPI support
- Target line display
- Responsive design

### RobotStatusGrid
- Multi-robot overview
- Status indicators
- Quick actions
- Detailed information panels

## Performance Optimizations

- React Query for efficient data fetching
- Component memoization
- Lazy loading for heavy components
- WebSocket connection management
- Optimized re-renders

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Follow the existing code style
2. Use TypeScript for all new code
3. Add proper error handling
4. Include unit tests for new features
5. Update documentation

## Security Considerations

- Input validation and sanitization
- XSS prevention
- Secure WebSocket connections
- Environment variable protection
- Content Security Policy (CSP)

## Troubleshooting

### Common Issues

1. **WebSocket Connection Issues**
   - Check backend service is running
   - Verify WebSocket URL in environment variables
   - Check network connectivity

2. **API Connection Errors**
   - Verify API base URL configuration
   - Check CORS settings on backend
   - Ensure backend services are accessible

3. **Performance Issues**
   - Check React Query cache settings
   - Verify WebSocket connection stability
   - Monitor browser console for errors

### Debug Mode

Enable debug mode by setting:
```bash
REACT_APP_DEBUG=true npm start
```

This will enable additional logging and debugging information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please contact the development team or create an issue in the project repository.
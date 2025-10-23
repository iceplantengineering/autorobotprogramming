import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CircularProgress, useTheme } from '@mui/material';
import { toast } from 'react-hot-toast';

import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Robots from './pages/Robots';
import Production from './pages/Production';
import Optimization from './pages/Optimization';
import Settings from './pages/Settings';
import NotFound from './pages/NotFound';

import { useWebSocket } from './hooks/useWebSocket';
import { useSystemStatus } from './hooks/useSystemStatus';
import { navigationItems } from './utils/navigation';
import { WebSocketMessage } from './types';

const App: React.FC = () => {
  const theme = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [systemReady, setSystemReady] = useState(false);

  // WebSocket connection for real-time updates
  const { lastMessage, connectionStatus } = useWebSocket('ws://localhost:8080/ws/dashboard');

  // System status monitoring
  const { systemHealth, isLoading: systemLoading } = useSystemStatus();

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const message: WebSocketMessage = JSON.parse(lastMessage.data);
        handleWebSocketMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    }
  }, [lastMessage]);

  const handleWebSocketMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'system_alert':
        toast.error(message.data.message, {
          duration: 5000,
          icon: '🚨',
        });
        break;
      case 'production_update':
        toast.success(message.data.message, {
          duration: 3000,
          icon: '✅',
        });
        break;
      case 'quality_alert':
        toast.error(message.data.message, {
          duration: 4000,
          icon: '⚠️',
        });
        break;
      case 'connection_established':
        console.log('WebSocket connected:', message.data.clientId);
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  // Check system readiness
  useEffect(() => {
    if (!systemLoading && systemHealth) {
      setSystemReady(true);

      // Show connection status toast
      if (connectionStatus === 'connected') {
        toast.success('Connected to production system', {
          icon: '🔗',
        });
      } else if (connectionStatus === 'error') {
        toast.error('Failed to connect to production system', {
          icon: '❌',
        });
      }
    }
  }, [systemLoading, systemHealth, connectionStatus]);

  // Determine connection status color
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return theme.palette.success.main;
      case 'connecting':
        return theme.palette.warning.main;
      case 'error':
      case 'disconnected':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  };

  if (systemLoading || !systemReady) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        bgcolor="background.default"
      >
        <Box textAlign="center">
          <CircularProgress size={60} thickness={4} />
          <Box mt={2} color="text.secondary">
            Initializing Robot Production Monitor...
          </Box>
          <Box mt={1} fontSize="body2" color="text.disabled">
            Connection status: {connectionStatus}
          </Box>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        navigationItems={navigationItems}
      />

      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <Header
          onMenuClick={() => setSidebarOpen(!sidebarOpen)}
          connectionStatus={connectionStatus}
          systemHealth={systemHealth}
        />

        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            bgcolor: 'background.default',
            overflow: 'auto',
          }}
        >
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/robots" element={<Robots />} />
            <Route path="/production" element={<Production />} />
            <Route path="/optimization" element={<Optimization />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Box>
      </Box>

      {/* Connection Status Indicator */}
      <Box
        sx={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 2,
          py: 1,
          bgcolor: 'background.paper',
          borderRadius: 2,
          boxShadow: 2,
          zIndex: 1000,
        }}
      >
        <Box
          sx={{
            width: 12,
            height: 12,
            borderRadius: '50%',
            bgcolor: getConnectionStatusColor(),
            animation: connectionStatus === 'connecting' ? 'pulse 2s infinite' : 'none',
          }}
        />
        <Box fontSize="body2" color="text.secondary">
          {connectionStatus}
        </Box>
      </Box>

      <style jsx>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </Box>
  );
};

export default App;
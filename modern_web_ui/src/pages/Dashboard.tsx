import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Alert,
  CircularProgress,
  useTheme,
} from '@mui/material';

import { useQuery } from 'react-query';

import MetricCard from '@/components/Metrics/MetricCard';
import AlertList from '@/components/Alerts/AlertList';
import KPITrendChart from '@/components/Charts/KPITrendChart';
import RobotStatusGrid from '@/components/Robots/RobotStatusGrid';
import ProductionSummary from '@/components/Production/ProductionSummary';

import apiService from '@/services/api';
import { DashboardResponse } from '@/services/api';
import { formatNumber, formatPercentage } from '@/utils/formatters';

const Dashboard: React.FC = () => {
  const theme = useTheme();

  // Fetch dashboard data
  const { data: dashboardData, isLoading, error } = useQuery<DashboardResponse>(
    'dashboard',
    apiService.getProductionDashboard,
    {
      refetchInterval: 30000, // Refresh every 30 seconds
      select: (response) => response.data,
    }
  );

  // Fetch current metrics
  const { data: metricsData } = useQuery(
    'currentMetrics',
    apiService.getCurrentMetrics,
    {
      refetchInterval: 10000, // Refresh every 10 seconds
      select: (response) => response.data,
    }
  );

  // Fetch active alerts
  const { data: alertsData } = useQuery(
    'activeAlerts',
    () => apiService.getActiveAlerts({ acknowledged: false }),
    {
      refetchInterval: 15000, // Refresh every 15 seconds
      select: (response) => response.data,
    }
  );

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error || !dashboardData) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Failed to load dashboard data. Please check your connection.
      </Alert>
    );
  }

  const { systemHealth, activeRobots, totalOrders, completedOrders, averageOEE, averageQuality } = dashboardData;

  return (
    <Box>
      {/* Page Header */}
      <Box mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Production Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Real-time monitoring of robot production systems
        </Typography>
      </Box>

      {/* System Health Alert */}
      {systemHealth.overall !== 'healthy' && (
        <Alert
          severity={systemHealth.overall === 'critical' ? 'error' : 'warning'}
          sx={{ mb: 3 }}
        >
          System health is {systemHealth.overall}. Some components may require attention.
        </Alert>
      )}

      {/* Key Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Robots"
            value={activeRobots}
            icon="🤖"
            color="primary"
            trend={{
              direction: 'stable',
              percentage: 0,
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Orders"
            value={totalOrders}
            icon="📋"
            color="info"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Completed"
            value={completedOrders}
            icon="✅"
            color="success"
            trend={{
              direction: 'up',
              percentage: totalOrders > 0 ? (completedOrders / totalOrders) * 100 : 0,
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Completion Rate"
            value={formatPercentage(totalOrders > 0 ? completedOrders / totalOrders : 0)}
            icon="📈"
            color={totalOrders > 0 && completedOrders / totalOrders > 0.8 ? 'success' : 'warning'}
          />
        </Grid>
      </Grid>

      {/* Performance Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={6}>
          <MetricCard
            title="Average OEE"
            value={formatPercentage(averageOEE)}
            target={85}
            icon="⚡"
            color="primary"
            trend={{
              direction: averageOEE > 0.8 ? 'up' : 'down',
              percentage: averageOEE * 100,
            }}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <MetricCard
            title="Average Quality"
            value={formatPercentage(averageQuality)}
            target={95}
            icon="🎯"
            color="success"
            trend={{
              direction: averageQuality > 0.9 ? 'stable' : 'down',
              percentage: averageQuality * 100,
            }}
          />
        </Grid>
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Robot Status */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Robot Status
              </Typography>
              <RobotStatusGrid />
            </CardContent>
          </Card>
        </Grid>

        {/* Active Alerts */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Alerts
              </Typography>
              <AlertList
                alerts={alertsData?.alerts || []}
                maxItems={5}
                showAcknowledgeButton={true}
                onAcknowledge={(alertId) => {
                  // Handle alert acknowledgment
                  console.log('Acknowledging alert:', alertId);
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Production Summary */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Production Summary
              </Typography>
              <ProductionSummary data={dashboardData} />
            </CardContent>
          </Card>
        </Grid>

        {/* OEE Trend Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                OEE Trend (Last 24 Hours)
              </Typography>
              <KPITrendChart
                kpiData={metricsData?.kpis?.oee ? [metricsData.kpis.oee] : []}
                title="Overall Equipment Effectiveness"
                height={200}
                showTarget={true}
                targetValue={85}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Additional Analytics Section */}
      <Grid container spacing={3} mt={2}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Performance Overview
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="primary">
                      {formatNumber(dashboardData?.recentActivity?.length || 0)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Recent Activities
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="success">
                      {formatNumber(Object.keys(dashboardData?.kpis || {}).length)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Active KPIs
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="warning">
                      {alertsData?.alerts?.length || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Active Alerts
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="info">
                      {systemHealth?.overall === 'healthy' ? '✅' : '⚠️'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      System Health
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
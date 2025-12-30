// ============================================================
// Main App Component with Routing
// ============================================================
import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import { MainLayout } from '@/components/Layout';

// Lazy load pages for better performance
const DashboardPage = lazy(() =>
  import('@/pages/DashboardPage').then((module) => ({ default: module.DashboardPage }))
);
const OntologyPage = lazy(() =>
  import('@/pages/OntologyPage').then((module) => ({ default: module.OntologyPage }))
);
const SPCPage = lazy(() =>
  import('@/pages/SPCPage').then((module) => ({ default: module.SPCPage }))
);
const AlertsPage = lazy(() =>
  import('@/pages/AlertsPage').then((module) => ({ default: module.AlertsPage }))
);
const TestPage = lazy(() =>
  import('@/pages/TestPage').then((module) => ({ default: module.TestPage }))
);
const AgentPage = lazy(() =>
  import('@/pages/AgentPage').then((module) => ({ default: module.AgentPage }))
);

// Loading fallback
const PageLoader: React.FC = () => (
  <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
    <div className="text-center">
      <div className="loader mx-auto mb-4" />
      <p className="text-slate-400 text-sm">Loading...</p>
    </div>
  </div>
);

// 404 Page
const NotFoundPage: React.FC = () => (
  <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
    <div className="text-center">
      <h1 className="text-6xl font-bold text-slate-600 mb-4">404</h1>
      <p className="text-xl text-slate-400 mb-6">Page not found</p>
      <a href="/" className="btn btn-primary">
        Go to Dashboard
      </a>
    </div>
  </div>
);

const App: React.FC = () => {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route
          index
          element={
            <Suspense fallback={<PageLoader />}>
              <DashboardPage />
            </Suspense>
          }
        />
        <Route
          path="/ontology"
          element={
            <Suspense fallback={<PageLoader />}>
              <OntologyPage />
            </Suspense>
          }
        />
        <Route
          path="/equipment"
          element={
            <Suspense fallback={<PageLoader />}>
              <DashboardPage />
            </Suspense>
          }
        />
        <Route
          path="/spc"
          element={
            <Suspense fallback={<PageLoader />}>
              <SPCPage />
            </Suspense>
          }
        />
        <Route
          path="/analytics"
          element={
            <Suspense fallback={<PageLoader />}>
              <SPCPage />
            </Suspense>
          }
        />
        <Route
          path="/alerts"
          element={
            <Suspense fallback={<PageLoader />}>
              <AlertsPage />
            </Suspense>
          }
        />
        <Route
          path="/test"
          element={
            <Suspense fallback={<PageLoader />}>
              <TestPage />
            </Suspense>
          }
        />
        <Route
          path="/agent"
          element={
            <Suspense fallback={<PageLoader />}>
              <AgentPage />
            </Suspense>
          }
        />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
};

export default App;

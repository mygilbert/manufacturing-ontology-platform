// ============================================================
// Ontology Graph Page - Enhanced Version
// ============================================================
import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  OntologyGraph,
  NodeDetail,
  FilterPanel,
  PathFinder,
  DiscoveredRelationships,
  LayoutOptions,
} from '@/components/OntologyGraph';
import type {
  NodeTypeFilter,
  RelationFilter,
  DiscoveredRelation,
  LayoutType,
} from '@/components/OntologyGraph';
import { api } from '@/services/api';
import type { GraphNode, GraphData } from '@/types';

// Node type metadata
const NODE_TYPE_META: Record<string, { label: string; color: string }> = {
  Equipment: { label: 'Equipment', color: '#3b82f6' },
  Lot: { label: 'Lot', color: '#10b981' },
  Wafer: { label: 'Wafer', color: '#6366f1' },
  Process: { label: 'Process', color: '#f59e0b' },
  Recipe: { label: 'Recipe', color: '#8b5cf6' },
  Measurement: { label: 'Measurement', color: '#06b6d4' },
  Alarm: { label: 'Alarm', color: '#ef4444' },
  Parameter: { label: 'Parameter', color: '#84cc16' },
};

// API response type for discovered relationships
interface ApiDiscoveredRelationship {
  id: string;
  source: string;
  target: string;
  relation_type: string;
  method: string;
  confidence: number;
  properties?: Record<string, unknown>;
  discovered_at: string;
  verification_status: string;
}

// Map API response to frontend type
const mapApiToDiscoveredRelation = (api: ApiDiscoveredRelationship): DiscoveredRelation => ({
  id: api.id,
  source: api.source,
  target: api.target,
  type: api.relation_type as 'correlation' | 'causality' | 'pattern',
  method: api.method,
  confidence: api.confidence,
  lag: api.properties?.optimal_lag as number | undefined,
  status: api.verification_status as 'pending' | 'verified' | 'rejected',
  properties: api.properties,
});

export const OntologyPage: React.FC = () => {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchType, setSearchType] = useState<string>('Equipment');
  const [searchId, setSearchId] = useState<string>('');

  // New state for enhanced features
  const [layout, setLayout] = useState<LayoutType>('force');
  const [showDiscoveredRelations, setShowDiscoveredRelations] = useState(false);
  const [discoveredRelations, setDiscoveredRelations] = useState<DiscoveredRelation[]>([]);
  const [loadingDiscoveries, setLoadingDiscoveries] = useState(false);
  const [highlightedPath, setHighlightedPath] = useState<string[]>([]);
  const [isSearchingPath, setIsSearchingPath] = useState(false);
  const [pathResult, setPathResult] = useState<{
    found: boolean;
    length: number;
    nodes: string[];
  } | null>(null);

  // Filter state
  const [nodeFilters, setNodeFilters] = useState<NodeTypeFilter[]>([]);
  const [relationFilters, setRelationFilters] = useState<RelationFilter[]>([]);

  // Compute visible types from filters
  const visibleNodeTypes = useMemo(() => {
    const visible = nodeFilters.filter((f) => f.visible).map((f) => f.type);
    return visible.length === nodeFilters.length ? undefined : new Set(visible);
  }, [nodeFilters]);

  const visibleRelationTypes = useMemo(() => {
    const visible = relationFilters.filter((f) => f.visible).map((f) => f.type);
    return visible.length === relationFilters.length ? undefined : new Set(visible);
  }, [relationFilters]);

  // Update filters when data changes
  useEffect(() => {
    // Count nodes by type
    const nodeCounts = new Map<string, number>();
    graphData.nodes.forEach((node) => {
      nodeCounts.set(node.type, (nodeCounts.get(node.type) || 0) + 1);
    });

    // Count edges by label
    const edgeCounts = new Map<string, number>();
    graphData.edges.forEach((edge) => {
      edgeCounts.set(edge.label, (edgeCounts.get(edge.label) || 0) + 1);
    });

    // Update node filters
    setNodeFilters((prev) => {
      const existingTypes = new Set(prev.map((f) => f.type));
      const newFilters: NodeTypeFilter[] = [];

      nodeCounts.forEach((count, type) => {
        const existing = prev.find((f) => f.type === type);
        const meta = NODE_TYPE_META[type] || { label: type, color: '#6b7280' };
        newFilters.push({
          type,
          label: meta.label,
          color: meta.color,
          visible: existing?.visible ?? true,
          count,
        });
      });

      return newFilters;
    });

    // Update relation filters
    setRelationFilters((prev) => {
      const newFilters: RelationFilter[] = [];
      edgeCounts.forEach((count, type) => {
        const existing = prev.find((f) => f.type === type);
        newFilters.push({
          type,
          visible: existing?.visible ?? true,
          count,
        });
      });
      return newFilters;
    });
  }, [graphData]);

  // Fetch discoveries from API
  useEffect(() => {
    const fetchDiscoveries = async () => {
      if (!showDiscoveredRelations) return;

      try {
        setLoadingDiscoveries(true);
        const response = await fetch('/api/analytics/discoveries');
        if (response.ok) {
          const data = await response.json();
          const mapped = data.discoveries.map(mapApiToDiscoveredRelation);
          setDiscoveredRelations(mapped);
        }
      } catch (error) {
        console.error('Failed to fetch discoveries:', error);
      } finally {
        setLoadingDiscoveries(false);
      }
    };

    fetchDiscoveries();
  }, [showDiscoveredRelations]);

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        const result = await api.traverseGraph({
          start_type: 'Equipment',
          start_id: 'EQP-ETCH-001',
          direction: 'both',
          depth: 2,
        });
        setGraphData(result);
      } catch (error) {
        console.error('Failed to load graph data:', error);
        // Use mock data
        setGraphData({
          nodes: [
            { id: 'EQP001', label: 'Etcher-01', type: 'Equipment', properties: { status: 'RUNNING', equipment_type: 'DRY_ETCH' } },
            { id: 'EQP002', label: 'CVD-01', type: 'Equipment', properties: { status: 'IDLE', equipment_type: 'CVD' } },
            { id: 'EQP003', label: 'Litho-01', type: 'Equipment', properties: { status: 'RUNNING', equipment_type: 'LITHO' } },
            { id: 'LOT001', label: 'LOT-20231201-001', type: 'Lot', properties: { status: 'IN_PROCESS', product_code: 'PROD-A' } },
            { id: 'LOT002', label: 'LOT-20231201-002', type: 'Lot', properties: { status: 'COMPLETED', product_code: 'PROD-B' } },
            { id: 'WFR001', label: 'WFR-001', type: 'Wafer', properties: { slot_no: 1, status: 'IN_PROCESS' } },
            { id: 'WFR002', label: 'WFR-002', type: 'Wafer', properties: { slot_no: 2, status: 'IN_PROCESS' } },
            { id: 'WFR003', label: 'WFR-003', type: 'Wafer', properties: { slot_no: 3, status: 'COMPLETED' } },
            { id: 'PRC001', label: 'Etch Process', type: 'Process', properties: { sequence: 1 } },
            { id: 'PRC002', label: 'CVD Process', type: 'Process', properties: { sequence: 2 } },
            { id: 'RCP001', label: 'ETCH-V2.1', type: 'Recipe', properties: { version: '2.1' } },
            { id: 'RCP002', label: 'CVD-V1.5', type: 'Recipe', properties: { version: '1.5' } },
            { id: 'ALM001', label: 'Temp High', type: 'Alarm', properties: { severity: 'WARNING' } },
            { id: 'ALM002', label: 'Pressure Low', type: 'Alarm', properties: { severity: 'CRITICAL' } },
            { id: 'MSR001', label: 'Temperature', type: 'Measurement', properties: { value: 350.5, unit: 'C' } },
          ],
          edges: [
            { id: 'e1', source: 'LOT001', target: 'EQP001', label: 'PROCESSED_AT', properties: { confidence: 1.0 } },
            { id: 'e2', source: 'LOT002', target: 'EQP002', label: 'PROCESSED_AT', properties: { confidence: 1.0 } },
            { id: 'e3', source: 'WFR001', target: 'LOT001', label: 'BELONGS_TO', properties: {} },
            { id: 'e4', source: 'WFR002', target: 'LOT001', label: 'BELONGS_TO', properties: {} },
            { id: 'e5', source: 'WFR003', target: 'LOT002', label: 'BELONGS_TO', properties: {} },
            { id: 'e6', source: 'PRC001', target: 'RCP001', label: 'USES', properties: {} },
            { id: 'e7', source: 'PRC002', target: 'RCP002', label: 'USES', properties: {} },
            { id: 'e8', source: 'EQP001', target: 'PRC001', label: 'EXECUTES', properties: {} },
            { id: 'e9', source: 'EQP002', target: 'PRC002', label: 'EXECUTES', properties: {} },
            { id: 'e10', source: 'EQP001', target: 'ALM001', label: 'GENERATES', properties: { confidence: 0.85 } },
            { id: 'e11', source: 'EQP001', target: 'ALM002', label: 'GENERATES', properties: { confidence: 0.92 } },
            { id: 'e12', source: 'EQP001', target: 'MSR001', label: 'MEASURES', properties: {} },
            { id: 'e13', source: 'MSR001', target: 'ALM001', label: 'TRIGGERS', properties: { confidence: 0.78 } },
          ],
        });
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, []);

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  const handleNodeDoubleClick = useCallback(async (node: GraphNode) => {
    try {
      setLoading(true);
      const result = await api.traverseGraph({
        start_type: node.type,
        start_id: node.id,
        direction: 'both',
        depth: 2,
      });

      setGraphData((prev) => {
        const existingNodeIds = new Set(prev.nodes.map((n) => n.id));
        const existingEdgeIds = new Set(prev.edges.map((e) => e.id));

        const newNodes = result.nodes.filter((n) => !existingNodeIds.has(n.id));
        const newEdges = result.edges.filter((e) => !existingEdgeIds.has(e.id));

        return {
          nodes: [...prev.nodes, ...newNodes],
          edges: [...prev.edges, ...newEdges],
        };
      });
    } catch (error) {
      console.error('Failed to expand node:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSearch = async () => {
    if (!searchId.trim()) return;

    try {
      setLoading(true);
      const result = await api.traverseGraph({
        start_type: searchType,
        start_id: searchId,
        direction: 'both',
        depth: 2,
      });
      setGraphData(result);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  // Path finding
  const handleFindPath = async (
    from: { type: string; id: string },
    to: { type: string; id: string }
  ) => {
    try {
      setIsSearchingPath(true);
      setPathResult(null);

      const result = await api.findPath({
        from_type: from.type,
        from_id: from.id,
        to_type: to.type,
        to_id: to.id,
        max_depth: 5,
      });

      if (result.found) {
        const nodeIds = result.nodes.map((n) => n.id);
        setHighlightedPath(nodeIds);
        setPathResult({
          found: true,
          length: result.length,
          nodes: nodeIds,
        });
      } else {
        setHighlightedPath([]);
        setPathResult({ found: false, length: 0, nodes: [] });
      }
    } catch (error) {
      console.error('Path finding failed:', error);
      // Demo fallback
      const demoPath = ['EQP001', 'ALM001'];
      setHighlightedPath(demoPath);
      setPathResult({
        found: true,
        length: 1,
        nodes: demoPath,
      });
    } finally {
      setIsSearchingPath(false);
    }
  };

  const handleClearPath = () => {
    setHighlightedPath([]);
    setPathResult(null);
  };

  // Filter handlers
  const handleNodeFilterChange = (type: string, visible: boolean) => {
    setNodeFilters((prev) =>
      prev.map((f) => (f.type === type ? { ...f, visible } : f))
    );
  };

  const handleRelationFilterChange = (type: string, visible: boolean) => {
    setRelationFilters((prev) =>
      prev.map((f) => (f.type === type ? { ...f, visible } : f))
    );
  };

  const handleSelectAll = () => {
    setNodeFilters((prev) => prev.map((f) => ({ ...f, visible: true })));
    setRelationFilters((prev) => prev.map((f) => ({ ...f, visible: true })));
  };

  const handleDeselectAll = () => {
    setNodeFilters((prev) => prev.map((f) => ({ ...f, visible: false })));
    setRelationFilters((prev) => prev.map((f) => ({ ...f, visible: false })));
  };

  // Discovered relations handlers
  const handleAddToGraph = (relation: DiscoveredRelation) => {
    const newEdge = {
      id: `discovered-${relation.id}`,
      source: relation.source,
      target: relation.target,
      label: relation.type.toUpperCase(),
      properties: {
        confidence: relation.confidence,
        method: relation.method,
        discovered: true,
      },
    };

    // Check if nodes exist, if not add them
    const sourceExists = graphData.nodes.some((n) => n.id === relation.source);
    const targetExists = graphData.nodes.some((n) => n.id === relation.target);

    const newNodes: GraphNode[] = [];
    if (!sourceExists) {
      newNodes.push({
        id: relation.source,
        label: relation.source,
        type: 'Parameter',
        properties: { discovered: true },
      });
    }
    if (!targetExists) {
      newNodes.push({
        id: relation.target,
        label: relation.target,
        type: 'Parameter',
        properties: { discovered: true },
      });
    }

    setGraphData((prev) => ({
      nodes: [...prev.nodes, ...newNodes],
      edges: [...prev.edges, newEdge],
    }));
  };

  const handleVerifyRelation = async (id: string) => {
    try {
      const response = await fetch(`/api/analytics/discoveries/${id}/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verified_by: 'user', notes: 'Verified via UI' }),
      });

      if (response.ok) {
        setDiscoveredRelations((prev) =>
          prev.map((r) => (r.id === id ? { ...r, status: 'verified' } : r))
        );
      }
    } catch (error) {
      console.error('Failed to verify relationship:', error);
    }
  };

  const handleRejectRelation = async (id: string) => {
    try {
      const response = await fetch(`/api/analytics/discoveries/${id}/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verified_by: 'user', notes: 'Rejected via UI' }),
      });

      if (response.ok) {
        setDiscoveredRelations((prev) =>
          prev.map((r) => (r.id === id ? { ...r, status: 'rejected' } : r))
        );
      }
    } catch (error) {
      console.error('Failed to reject relationship:', error);
    }
  };

  const availableNodeTypes = Object.keys(NODE_TYPE_META);

  return (
    <div className="h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Ontology Graph</h1>
          <p className="text-slate-400 text-sm mt-1">
            Explore manufacturing relationships ({graphData.nodes.length} nodes, {graphData.edges.length} edges)
          </p>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-2">
          {/* Search */}
          <select
            value={searchType}
            onChange={(e) => setSearchType(e.target.value)}
            className="select w-32"
          >
            {availableNodeTypes.map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          <input
            type="text"
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
            placeholder="Enter ID..."
            className="input w-48"
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button onClick={handleSearch} className="btn btn-primary">
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Search
          </button>

          {/* Discovered Relations Toggle */}
          <button
            onClick={() => setShowDiscoveredRelations(!showDiscoveredRelations)}
            className={`btn ${showDiscoveredRelations ? 'btn-primary' : 'btn-secondary'}`}
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Discoveries
          </button>

          <button
            onClick={() => setGraphData({ nodes: [], edges: [] })}
            className="btn btn-secondary"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Graph Container */}
      <div className="relative h-[calc(100%-4rem)] card overflow-hidden">
        {loading && (
          <div className="absolute inset-0 bg-slate-900/50 flex items-center justify-center z-30">
            <div className="loader" />
          </div>
        )}

        {/* Filter Panel */}
        <FilterPanel
          nodeFilters={nodeFilters}
          relationFilters={relationFilters}
          onNodeFilterChange={handleNodeFilterChange}
          onRelationFilterChange={handleRelationFilterChange}
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
        />

        {/* Path Finder */}
        <PathFinder
          nodeTypes={availableNodeTypes}
          onFindPath={handleFindPath}
          onClearPath={handleClearPath}
          isSearching={isSearchingPath}
          pathResult={pathResult}
        />

        {/* Main Graph */}
        <OntologyGraph
          data={graphData}
          layout={layout}
          highlightedPath={highlightedPath}
          selectedNodeId={selectedNode?.id || null}
          visibleNodeTypes={visibleNodeTypes}
          visibleRelationTypes={visibleRelationTypes}
          onNodeClick={handleNodeClick}
          onNodeDoubleClick={handleNodeDoubleClick}
          onBackgroundClick={() => setSelectedNode(null)}
        />

        {/* Layout Options */}
        <LayoutOptions
          currentLayout={layout}
          onLayoutChange={setLayout}
        />
      </div>

      {/* Node Detail Panel */}
      <NodeDetail
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onExpand={handleNodeDoubleClick}
      />

      {/* Discovered Relationships Panel */}
      {showDiscoveredRelations && (
        <DiscoveredRelationships
          relations={discoveredRelations}
          onAddToGraph={handleAddToGraph}
          onVerify={handleVerifyRelation}
          onReject={handleRejectRelation}
          onClose={() => setShowDiscoveredRelations(false)}
        />
      )}
    </div>
  );
};

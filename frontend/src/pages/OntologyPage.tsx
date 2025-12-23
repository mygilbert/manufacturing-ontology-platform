// ============================================================
// Ontology Graph Page
// ============================================================
import React, { useState, useCallback, useEffect } from 'react';
import { OntologyGraph, NodeDetail } from '@/components/OntologyGraph';
import { api } from '@/services/api';
import type { GraphNode, GraphData } from '@/types';

export const OntologyPage: React.FC = () => {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchType, setSearchType] = useState<string>('Equipment');
  const [searchId, setSearchId] = useState<string>('');

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        // Try to fetch from API
        const result = await api.traverseGraph({
          start_type: 'Equipment',
          start_id: 'EQP001',
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
            { id: 'LOT001', label: 'LOT-20231201-001', type: 'Lot', properties: { status: 'IN_PROCESS', product_code: 'PROD-A' } },
            { id: 'LOT002', label: 'LOT-20231201-002', type: 'Lot', properties: { status: 'COMPLETED', product_code: 'PROD-B' } },
            { id: 'WFR001', label: 'WFR-001', type: 'Wafer', properties: { slot_no: 1, status: 'IN_PROCESS' } },
            { id: 'WFR002', label: 'WFR-002', type: 'Wafer', properties: { slot_no: 2, status: 'IN_PROCESS' } },
            { id: 'PRC001', label: 'Etch Process', type: 'Process', properties: { sequence: 1 } },
            { id: 'RCP001', label: 'ETCH-V2.1', type: 'Recipe', properties: { version: '2.1' } },
            { id: 'ALM001', label: 'Temp High', type: 'Alarm', properties: { severity: 'WARNING' } },
          ],
          edges: [
            { id: 'e1', source: 'LOT001', target: 'EQP001', label: 'PROCESSED_AT', properties: {} },
            { id: 'e2', source: 'LOT002', target: 'EQP002', label: 'PROCESSED_AT', properties: {} },
            { id: 'e3', source: 'WFR001', target: 'LOT001', label: 'BELONGS_TO', properties: {} },
            { id: 'e4', source: 'WFR002', target: 'LOT001', label: 'BELONGS_TO', properties: {} },
            { id: 'e5', source: 'PRC001', target: 'RCP001', label: 'USES', properties: {} },
            { id: 'e6', source: 'EQP001', target: 'PRC001', label: 'EXECUTES', properties: {} },
            { id: 'e7', source: 'EQP001', target: 'ALM001', label: 'GENERATES', properties: {} },
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

      // Merge new data with existing
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

  return (
    <div className="h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Ontology Graph</h1>
          <p className="text-slate-400 text-sm mt-1">Explore manufacturing relationships</p>
        </div>

        {/* Search */}
        <div className="flex items-center space-x-2">
          <select
            value={searchType}
            onChange={(e) => setSearchType(e.target.value)}
            className="select w-32"
          >
            <option value="Equipment">Equipment</option>
            <option value="Lot">Lot</option>
            <option value="Wafer">Wafer</option>
            <option value="Process">Process</option>
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
          <div className="absolute inset-0 bg-slate-900/50 flex items-center justify-center z-10">
            <div className="loader" />
          </div>
        )}
        <OntologyGraph
          data={graphData}
          onNodeClick={handleNodeClick}
          onNodeDoubleClick={handleNodeDoubleClick}
        />
      </div>

      {/* Node Detail Panel */}
      <NodeDetail
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onExpand={handleNodeDoubleClick}
      />
    </div>
  );
};

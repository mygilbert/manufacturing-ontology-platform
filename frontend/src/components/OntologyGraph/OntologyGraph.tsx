// ============================================================
// Ontology Graph Visualization with D3.js
// ============================================================
import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import type { GraphNode, GraphEdge, GraphData } from '@/types';
import type { LayoutType } from './LayoutOptions';

interface OntologyGraphProps {
  data: GraphData;
  width?: number;
  height?: number;
  layout?: LayoutType;
  highlightedPath?: string[];
  selectedNodeId?: string | null;
  visibleNodeTypes?: Set<string>;
  visibleRelationTypes?: Set<string>;
  onNodeClick?: (node: GraphNode) => void;
  onNodeDoubleClick?: (node: GraphNode) => void;
  onBackgroundClick?: () => void;
}

// Node type color mapping
const NODE_COLORS: Record<string, string> = {
  Equipment: '#3b82f6',    // Blue
  Lot: '#10b981',          // Green
  Wafer: '#6366f1',        // Indigo
  Process: '#f59e0b',      // Amber
  Recipe: '#8b5cf6',       // Purple
  Measurement: '#06b6d4',  // Cyan
  Alarm: '#ef4444',        // Red
  Parameter: '#84cc16',    // Lime
  default: '#6b7280',      // Gray
};

// Node type icons (simplified path data)
const NODE_ICONS: Record<string, string> = {
  Equipment: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z',
  Lot: 'M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4',
  Wafer: 'M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  Process: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547',
  Recipe: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z',
  Alarm: 'M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9',
};

// Edge strength color mapping
const getEdgeColor = (strength?: number): string => {
  if (strength === undefined) return '#475569';
  if (strength >= 0.8) return '#10b981'; // Green - strong
  if (strength >= 0.6) return '#f59e0b'; // Amber - medium
  if (strength >= 0.4) return '#6b7280'; // Gray - weak
  return '#374151'; // Dark gray - very weak
};

const getEdgeWidth = (strength?: number): number => {
  if (strength === undefined) return 2;
  return 1 + strength * 4; // 1-5 range
};

export const OntologyGraph: React.FC<OntologyGraphProps> = ({
  data,
  width = 1000,
  height = 700,
  layout = 'force',
  highlightedPath = [],
  selectedNodeId = null,
  visibleNodeTypes,
  visibleRelationTypes,
  onNodeClick,
  onNodeDoubleClick,
  onBackgroundClick,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    content: GraphNode | null;
  }>({ visible: false, x: 0, y: 0, content: null });

  const [dimensions, setDimensions] = useState({ width, height });

  // Filter data based on visible types
  const filteredData = useMemo(() => {
    let nodes = data.nodes;
    let edges = data.edges;

    if (visibleNodeTypes && visibleNodeTypes.size > 0) {
      nodes = nodes.filter((n) => visibleNodeTypes.has(n.type));
      const nodeIds = new Set(nodes.map((n) => n.id));
      edges = edges.filter((e) => {
        const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
        const targetId = typeof e.target === 'string' ? e.target : e.target.id;
        return nodeIds.has(sourceId) && nodeIds.has(targetId);
      });
    }

    if (visibleRelationTypes && visibleRelationTypes.size > 0) {
      edges = edges.filter((e) => visibleRelationTypes.has(e.label));
    }

    return { nodes, edges };
  }, [data, visibleNodeTypes, visibleRelationTypes]);

  // Highlighted path set for fast lookup
  const highlightedSet = useMemo(() => new Set(highlightedPath), [highlightedPath]);

  // Find connected nodes and edges for selected node
  const { connectedNodes, connectedEdges } = useMemo(() => {
    if (!selectedNodeId) {
      return { connectedNodes: new Set<string>(), connectedEdges: new Set<string>() };
    }

    const nodes = new Set<string>([selectedNodeId]);
    const edges = new Set<string>();

    filteredData.edges.forEach((edge) => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;

      if (sourceId === selectedNodeId || targetId === selectedNodeId) {
        edges.add(edge.id);
        nodes.add(sourceId);
        nodes.add(targetId);
      }
    });

    return { connectedNodes: nodes, connectedEdges: edges };
  }, [selectedNodeId, filteredData.edges]);

  // Handle resize
  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height: Math.max(height, 500) });
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  // Main D3 rendering
  useEffect(() => {
    if (!svgRef.current || !filteredData.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { width, height } = dimensions;

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Background click to deselect
    svg.on('click', (event) => {
      // Only trigger if clicking on the SVG background itself
      if (event.target === svgRef.current) {
        onBackgroundClick?.();
      }
    });

    // Main container for zoom
    const container = svg.append('g');

    // Defs for markers and gradients
    const defs = svg.append('defs');

    // Arrow marker for edges
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#64748b');

    // Highlighted arrow marker
    defs.append('marker')
      .attr('id', 'arrowhead-highlighted')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#3b82f6');

    // Glow filter for highlighted nodes
    const filter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');

    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Prepare links with source/target as objects
    const links = filteredData.edges.map((edge) => ({
      ...edge,
      source: typeof edge.source === 'string' ? edge.source : edge.source.id,
      target: typeof edge.target === 'string' ? edge.target : edge.target.id,
    }));

    // Apply layout
    let simulation: d3.Simulation<d3.SimulationNodeDatum, undefined>;

    if (layout === 'force') {
      // Clear any fixed positions from other layouts
      filteredData.nodes.forEach((node) => {
        node.fx = null;
        node.fy = null;
      });

      simulation = d3.forceSimulation(filteredData.nodes as d3.SimulationNodeDatum[])
        .force('link', d3.forceLink(links)
          .id((d: any) => d.id)
          .distance(150)
          .strength(0.5))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(50))
        .alpha(1) // Force restart with full energy
        .alphaDecay(0.02); // Slower decay for smoother animation
    } else if (layout === 'hierarchical') {
      // Group nodes by type for hierarchical layout
      const typeGroups = new Map<string, GraphNode[]>();
      filteredData.nodes.forEach((node) => {
        if (!typeGroups.has(node.type)) {
          typeGroups.set(node.type, []);
        }
        typeGroups.get(node.type)!.push(node);
      });

      const types = Array.from(typeGroups.keys());
      const layerHeight = height / (types.length + 1);

      types.forEach((type, layerIndex) => {
        const nodesInLayer = typeGroups.get(type)!;
        const layerWidth = width / (nodesInLayer.length + 1);
        nodesInLayer.forEach((node, nodeIndex) => {
          node.x = layerWidth * (nodeIndex + 1);
          node.y = layerHeight * (layerIndex + 1);
          node.fx = node.x;
          node.fy = node.y;
        });
      });

      simulation = d3.forceSimulation(filteredData.nodes as d3.SimulationNodeDatum[])
        .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100));
    } else if (layout === 'radial') {
      // Radial layout
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) / 3;

      filteredData.nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / filteredData.nodes.length;
        node.x = centerX + radius * Math.cos(angle);
        node.y = centerY + radius * Math.sin(angle);
      });

      simulation = d3.forceSimulation(filteredData.nodes as d3.SimulationNodeDatum[])
        .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-100))
        .force('collision', d3.forceCollide().radius(40));
    } else {
      // Grid layout
      const cols = Math.ceil(Math.sqrt(filteredData.nodes.length));
      const cellWidth = width / (cols + 1);
      const cellHeight = height / (Math.ceil(filteredData.nodes.length / cols) + 1);

      filteredData.nodes.forEach((node, i) => {
        node.x = cellWidth * ((i % cols) + 1);
        node.y = cellHeight * (Math.floor(i / cols) + 1);
      });

      simulation = d3.forceSimulation(filteredData.nodes as d3.SimulationNodeDatum[])
        .force('link', d3.forceLink(links).id((d: any) => d.id));
    }

    // Selected node edge marker (cyan/teal color)
    defs.append('marker')
      .attr('id', 'arrowhead-selected')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#06b6d4'); // Cyan

    // Draw links
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'graph-link')
      .attr('stroke', (d: any) => {
        const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
        const targetId = typeof d.target === 'string' ? d.target : d.target.id;

        // Selected node connections - cyan color
        if (connectedEdges.has(d.id)) {
          return '#06b6d4'; // Cyan
        }
        // Path highlighting - blue color
        if (highlightedSet.has(sourceId) && highlightedSet.has(targetId)) {
          return '#3b82f6';
        }
        const strength = d.properties?.confidence || d.properties?.strength;
        return getEdgeColor(strength as number);
      })
      .attr('stroke-width', (d: any) => {
        const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
        const targetId = typeof d.target === 'string' ? d.target : d.target.id;

        // Selected node connections - thicker
        if (connectedEdges.has(d.id)) {
          return 4;
        }
        if (highlightedSet.has(sourceId) && highlightedSet.has(targetId)) {
          return 4;
        }
        const strength = d.properties?.confidence || d.properties?.strength;
        return getEdgeWidth(strength as number);
      })
      .attr('stroke-opacity', (d: any) => {
        // If a node is selected, dim non-connected edges
        if (selectedNodeId) {
          return connectedEdges.has(d.id) ? 1 : 0.15;
        }
        if (highlightedSet.size === 0) return 1;
        const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
        const targetId = typeof d.target === 'string' ? d.target : d.target.id;
        return (highlightedSet.has(sourceId) && highlightedSet.has(targetId)) ? 1 : 0.3;
      })
      .attr('marker-end', (d: any) => {
        const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
        const targetId = typeof d.target === 'string' ? d.target : d.target.id;

        if (connectedEdges.has(d.id)) {
          return 'url(#arrowhead-selected)';
        }
        return (highlightedSet.has(sourceId) && highlightedSet.has(targetId))
          ? 'url(#arrowhead-highlighted)'
          : 'url(#arrowhead)';
      });

    // Draw link labels
    const linkLabel = container.append('g')
      .attr('class', 'link-labels')
      .selectAll('text')
      .data(links)
      .enter()
      .append('text')
      .attr('class', 'graph-link-label')
      .attr('font-size', '10px')
      .attr('fill', '#94a3b8')
      .attr('text-anchor', 'middle')
      .text((d) => d.label);

    // Draw nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(filteredData.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .style('opacity', (d) => {
        // Selected node highlighting
        if (selectedNodeId) {
          return connectedNodes.has(d.id) ? 1 : 0.2;
        }
        // Path highlighting
        if (highlightedSet.size === 0) return 1;
        return highlightedSet.has(d.id) ? 1 : 0.3;
      })
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          if (layout === 'force') {
            d.fx = null;
            d.fy = null;
          }
        }) as any);

    // Node circles
    node.append('circle')
      .attr('r', (d) => {
        if (d.id === selectedNodeId) return 38;
        if (highlightedSet.has(d.id) || connectedNodes.has(d.id)) return 35;
        return 30;
      })
      .attr('fill', (d) => NODE_COLORS[d.type] || NODE_COLORS.default)
      .attr('stroke', (d) => {
        if (d.id === selectedNodeId) return '#06b6d4'; // Cyan for selected
        if (connectedNodes.has(d.id) && d.id !== selectedNodeId) return '#22d3ee'; // Light cyan for connected
        if (highlightedSet.has(d.id)) return '#3b82f6';
        return '#1e293b';
      })
      .attr('stroke-width', (d) => {
        if (d.id === selectedNodeId) return 5;
        if (connectedNodes.has(d.id) || highlightedSet.has(d.id)) return 4;
        return 3;
      })
      .attr('filter', (d) => {
        if (d.id === selectedNodeId || highlightedSet.has(d.id)) return 'url(#glow)';
        return null;
      })
      .on('mouseover', function (event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 35)
          .attr('stroke', '#ffffff')
          .attr('stroke-width', 4);

        setTooltip({
          visible: true,
          x: event.pageX,
          y: event.pageY,
          content: d,
        });
      })
      .on('mouseout', function () {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 30)
          .attr('stroke', '#1e293b')
          .attr('stroke-width', 3);

        setTooltip({ visible: false, x: 0, y: 0, content: null });
      })
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick?.(d);
      })
      .on('dblclick', (event, d) => {
        event.stopPropagation();
        onNodeDoubleClick?.(d);
      });

    // Node icons (simplified as text)
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', 'white')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('pointer-events', 'none')
      .text((d) => d.type.charAt(0));

    // Node labels
    node.append('text')
      .attr('dy', 50)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e2e8f0')
      .attr('font-size', '12px')
      .attr('font-weight', '500')
      .attr('pointer-events', 'none')
      .text((d) => d.label || d.id);

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      linkLabel
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [filteredData, dimensions, layout, highlightedSet, selectedNodeId, connectedNodes, connectedEdges, onNodeClick, onNodeDoubleClick, onBackgroundClick]);

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[500px]">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="bg-slate-900 rounded-lg"
      />

      {/* Tooltip */}
      {tooltip.visible && tooltip.content && (
        <div
          className="tooltip animate-fade-in"
          style={{
            left: tooltip.x + 10,
            top: tooltip.y + 10,
          }}
        >
          <div className="font-semibold text-white">{tooltip.content.label}</div>
          <div className="text-xs text-slate-400 mt-1">Type: {tooltip.content.type}</div>
          <div className="text-xs text-slate-400">ID: {tooltip.content.id}</div>
          {Object.entries(tooltip.content.properties).slice(0, 3).map(([key, value]) => (
            <div key={key} className="text-xs text-slate-400">
              {key}: {String(value)}
            </div>
          ))}
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-slate-800/90 backdrop-blur-sm rounded-lg p-3 border border-slate-700">
        <div className="text-xs font-semibold text-slate-400 mb-2">Node Types</div>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(NODE_COLORS).filter(([k]) => k !== 'default').map(([type, color]) => (
            <div key={type} className="flex items-center space-x-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs text-slate-300">{type}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col space-y-2">
        <button
          onClick={() => {
            const svg = d3.select(svgRef.current);
            svg.transition().duration(500).call(
              d3.zoom<SVGSVGElement, unknown>().transform as any,
              d3.zoomIdentity
            );
          }}
          className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          title="Reset Zoom"
        >
          <svg className="w-5 h-5 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>
    </div>
  );
};

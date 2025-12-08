import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { IconButton, ButtonGroup, Tooltip, Whisper } from 'rsuite';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearchPlus, faSearchMinus, faExpand, faCompress, faMaximize } from '@fortawesome/free-solid-svg-icons';

const Graph = ({ data }) => {
  const svgRef = useRef(null);
  const zoomRef = useRef(null);
  const graphContainerRef = useRef(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isFullScreen, setIsFullScreen] = useState(false);

  // Zoom functions
  const handleZoomIn = () => {
    if (zoomRef.current) {
      const newZoomLevel = Math.min(zoomLevel * 1.5, 8);
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.scaleBy, 1.5);
      setZoomLevel(newZoomLevel);
    }
  };

  const handleZoomOut = () => {
    if (zoomRef.current) {
      const newZoomLevel = Math.max(zoomLevel / 1.5, 0.1);
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.scaleBy, 1/1.5);
      setZoomLevel(newZoomLevel);
    }
  };

  const handleResetZoom = () => {
    if (zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(500)
        .call(zoomRef.current.transform, d3.zoomIdentity);
      setZoomLevel(1);
    }
  };

  // Full screen toggle function
  const toggleFullScreen = () => {
    if (!document.fullscreenElement && !document.mozFullScreenElement &&
        !document.webkitFullscreenElement && !document.msFullscreenElement) {
      // Enter full screen
      const container = graphContainerRef.current;
      if (container.requestFullscreen) {
        container.requestFullscreen();
      } else if (container.msRequestFullscreen) {
        container.msRequestFullscreen();
      } else if (container.mozRequestFullScreen) {
        container.mozRequestFullScreen();
      } else if (container.webkitRequestFullscreen) {
        container.webkitRequestFullscreen();
      }
      setIsFullScreen(true);
    } else {
      // Exit full screen
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
      } else if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen();
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      }
      setIsFullScreen(false);
    }
  };

  // Listen for full screen change events
  useEffect(() => {
    const handleFullScreenChange = () => {
      setIsFullScreen(
        document.fullscreenElement ||
        document.mozFullScreenElement ||
        document.webkitFullscreenElement ||
        document.msFullscreenElement
      );
    };

    document.addEventListener('fullscreenchange', handleFullScreenChange);
    document.addEventListener('mozfullscreenchange', handleFullScreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullScreenChange);
    document.addEventListener('MSFullscreenChange', handleFullScreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullScreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullScreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullScreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullScreenChange);
    };
  }, []);

  useEffect(() => {
    if (!data || !data.nodes || !data.edges || data.nodes.length === 0) {
      return;
    }

    // Clear previous graph
    d3.select(svgRef.current).selectAll('*').remove();

    // Set up SVG dimensions
    const width = 800;
    const height = 600;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height]);

    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(50));

    // Create a group for the graph
    const g = svg.append('g');

    // Create zoom behavior
    const zoom = d3.zoom()
      .extent([[0, 0], [width, height]])
      .scaleExtent([0.1, 8])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoomLevel(event.transform.k);
      });

    // Store zoom behavior in ref for external control
    zoomRef.current = zoom;

    // Add zoom behavior to SVG
    svg.call(zoom);

    // Define node colors based on type
    const color = d3.scaleOrdinal()
      .domain(['person', 'organization', 'event', 'geo', 'category', 'Unknown'])
      .range(['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', '#b07aa1']);

    // Create links
    const link = g.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(data.edges)
      .join('line')
      .attr('stroke-width', d => Math.sqrt(d.weight || 1));

    // Create nodes
    const node = g.append('g')
      .selectAll('.node')
      .data(data.nodes)
      .join('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add circles to nodes
    node.append('circle')
      .attr('r', 10)
      .attr('fill', d => color(d.type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);

    // Add labels to nodes
    node.append('text')
      .attr('dx', 12)
      .attr('dy', '.35em')
      .text(d => d.label)
      .attr('fill', '#fff');

    // Add tooltips
    node.append('title')
      .text(d => `${d.label}\nType: ${d.type}\n${d.description || ''}`);

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [data]);

  return (
    <div className={`graph-container ${isFullScreen ? 'fullscreen' : ''}`} ref={graphContainerRef}>
      <div className="graph-controls">
        <ButtonGroup>
          <Whisper placement="top" trigger="hover" speaker={<Tooltip>Zoom In</Tooltip>}>
            <IconButton
              icon={<FontAwesomeIcon icon={faSearchPlus} />}
              onClick={handleZoomIn}
              disabled={zoomLevel >= 8}
              appearance="subtle"
              size="sm"
            />
          </Whisper>

          <Whisper placement="top" trigger="hover" speaker={<Tooltip>Zoom Out</Tooltip>}>
            <IconButton
              icon={<FontAwesomeIcon icon={faSearchMinus} />}
              onClick={handleZoomOut}
              disabled={zoomLevel <= 0.1}
              appearance="subtle"
              size="sm"
            />
          </Whisper>

          <Whisper placement="top" trigger="hover" speaker={<Tooltip>Reset Zoom</Tooltip>}>
            <IconButton
              icon={<FontAwesomeIcon icon={faExpand} />}
              onClick={handleResetZoom}
              appearance="subtle"
              size="sm"
            />
          </Whisper>

          <Whisper placement="top" trigger="hover" speaker={<Tooltip>{isFullScreen ? 'Exit Full Screen' : 'Full Screen'}</Tooltip>}>
            <IconButton
              icon={<FontAwesomeIcon icon={isFullScreen ? faCompress : faMaximize} />}
              onClick={toggleFullScreen}
              appearance="subtle"
              size="sm"
            />
          </Whisper>
        </ButtonGroup>

        <div className="zoom-level">
          {Math.round(zoomLevel * 100)}%
        </div>
      </div>

      <svg ref={svgRef} className="knowledge-graph"></svg>
    </div>
  );
};

export default Graph;

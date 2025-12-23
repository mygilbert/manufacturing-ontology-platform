// ============================================================
// WebSocket Hook for Real-time Updates
// ============================================================
import { useEffect, useCallback, useState } from 'react';
import { wsService } from '@/services/websocket';
import type { WebSocketMessage, WebSocketMessageType } from '@/types';

interface UseWebSocketOptions {
  channels?: string[];
  autoConnect?: boolean;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { channels = [], autoConnect = true } = options;
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  useEffect(() => {
    if (autoConnect) {
      wsService.connect();
    }

    const unsubConnect = wsService.onConnect(() => {
      setIsConnected(true);
    });

    const unsubDisconnect = wsService.onDisconnect(() => {
      setIsConnected(false);
    });

    const unsubMessage = wsService.onMessage('all', (message) => {
      setLastMessage(message);
    });

    // Subscribe to channels
    channels.forEach((channel) => {
      wsService.subscribe(channel);
    });

    return () => {
      unsubConnect();
      unsubDisconnect();
      unsubMessage();
      channels.forEach((channel) => {
        wsService.unsubscribe(channel);
      });
    };
  }, [autoConnect, channels.join(',')]);

  const subscribe = useCallback((channel: string) => {
    wsService.subscribe(channel);
  }, []);

  const unsubscribe = useCallback((channel: string) => {
    wsService.unsubscribe(channel);
  }, []);

  return {
    isConnected,
    lastMessage,
    subscribe,
    unsubscribe,
  };
}

export function useWebSocketMessage<T = unknown>(
  type: WebSocketMessageType,
  handler: (data: T, message: WebSocketMessage<T>) => void
) {
  useEffect(() => {
    const unsubscribe = wsService.onMessage(type, (message) => {
      handler(message.data as T, message as WebSocketMessage<T>);
    });

    return unsubscribe;
  }, [type, handler]);
}

// ============================================================
// Agent Page - FDC Analysis Agent Interface
// ============================================================
import React from 'react';
import { AgentChat } from '@/components/AgentChat';

export const AgentPage: React.FC = () => {
  return (
    <div className="h-[calc(100vh-8rem)]">
      <AgentChat className="h-full" />
    </div>
  );
};

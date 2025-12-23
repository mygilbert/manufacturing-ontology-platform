// ============================================================
// Header Component
// ============================================================
import React, { useState } from 'react';
import { useStore, useWebSocket } from '@/hooks';
import classNames from 'classnames';

export const Header: React.FC = () => {
  const { sidebarOpen, toggleSidebar, unreadCount, clearUnread, activeAlarms } = useStore();
  const { isConnected } = useWebSocket();
  const [showNotifications, setShowNotifications] = useState(false);

  return (
    <header
      className={classNames(
        'fixed top-0 right-0 z-30 h-16 bg-slate-800/95 backdrop-blur-sm border-b border-slate-700 transition-all duration-300',
        sidebarOpen ? 'left-64' : 'left-20'
      )}
    >
      <div className="flex items-center justify-between h-full px-6">
        {/* Left Section */}
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleSidebar}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            <div
              className={classNames(
                'w-2 h-2 rounded-full',
                isConnected ? 'bg-emerald-500' : 'bg-red-500 animate-pulse'
              )}
            />
            <span className="text-sm text-slate-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center space-x-4">
          {/* Search */}
          <div className="relative">
            <input
              type="text"
              placeholder="Search..."
              className="w-64 px-4 py-2 pl-10 bg-slate-700 border border-slate-600 rounded-lg text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <svg
              className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>

          {/* Notifications */}
          <div className="relative">
            <button
              onClick={() => {
                setShowNotifications(!showNotifications);
                if (unreadCount > 0) clearUnread();
              }}
              className="relative p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
              </svg>
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs text-white flex items-center justify-center">
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </button>

            {/* Notification Dropdown */}
            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden">
                <div className="px-4 py-3 border-b border-slate-700">
                  <h3 className="font-semibold text-white">Notifications</h3>
                </div>
                <div className="max-h-96 overflow-y-auto">
                  {activeAlarms.length === 0 ? (
                    <div className="px-4 py-8 text-center text-slate-400">
                      No new notifications
                    </div>
                  ) : (
                    activeAlarms.slice(0, 5).map((alarm) => (
                      <div
                        key={alarm.alarm_id}
                        className="px-4 py-3 border-b border-slate-700/50 hover:bg-slate-700/50"
                      >
                        <div className="flex items-start space-x-3">
                          <div
                            className={classNames(
                              'w-2 h-2 mt-2 rounded-full',
                              alarm.severity === 'CRITICAL' && 'bg-red-500',
                              alarm.severity === 'WARNING' && 'bg-amber-500',
                              alarm.severity === 'INFO' && 'bg-blue-500'
                            )}
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm text-white truncate">
                              {alarm.message || alarm.alarm_code}
                            </p>
                            <p className="text-xs text-slate-400">
                              {alarm.equipment_id} - {new Date(alarm.occurred_at).toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                <div className="px-4 py-2 border-t border-slate-700 bg-slate-700/30">
                  <a href="/alerts" className="text-sm text-blue-400 hover:text-blue-300">
                    View all alerts
                  </a>
                </div>
              </div>
            )}
          </div>

          {/* User Menu */}
          <button className="flex items-center space-x-2 p-2 rounded-lg hover:bg-slate-700 transition-colors">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-medium">
              U
            </div>
          </button>
        </div>
      </div>
    </header>
  );
};

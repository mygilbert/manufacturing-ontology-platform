// ============================================================
// Alert System Test Page
// E2E 테스트 및 디버깅용 페이지
// ============================================================
import React, { useState, useEffect, useCallback } from 'react';
import { alertSound } from '@/utils/alertSound';
import { wsService } from '@/services/websocket';
import { useStore } from '@/hooks';
import { format } from 'date-fns';

interface LogEntry {
  id: number;
  time: string;
  type: 'info' | 'success' | 'error' | 'alert';
  message: string;
}

export const TestPage: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');

  const { activeAlarms, recentAnomalies, addAlarm, addAnomaly } = useStore();

  // 로그 추가 함수
  const addLog = useCallback((type: LogEntry['type'], message: string) => {
    setLogs((prev) => [
      {
        id: Date.now(),
        time: format(new Date(), 'HH:mm:ss.SSS'),
        type,
        message,
      },
      ...prev.slice(0, 99),
    ]);
  }, []);

  // WebSocket 연결 관리
  useEffect(() => {
    addLog('info', 'WebSocket 연결 시도 중...');

    const unsubConnect = wsService.onConnect(() => {
      setIsConnected(true);
      addLog('success', 'WebSocket 연결 성공!');
      if (soundEnabled) alertSound.playConnected();

      // 채널 구독
      wsService.subscribe('alerts');
      wsService.subscribe('anomalies');
      addLog('info', '채널 구독: alerts, anomalies');
    });

    const unsubDisconnect = wsService.onDisconnect(() => {
      setIsConnected(false);
      addLog('error', 'WebSocket 연결 끊김');
      if (soundEnabled) alertSound.playDisconnected();
    });

    // 알람 메시지 핸들러
    const unsubAlert = wsService.onMessage('alert', (message: any) => {
      addLog('alert', `🔔 알람 수신: [${message.severity}] ${message.message}`);

      // 스토어에 추가
      addAlarm({
        alarm_id: `alarm_${Date.now()}`,
        equipment_id: message.equipment_id || 'UNKNOWN',
        severity: message.severity || 'INFO',
        message: message.message,
        occurred_at: message.timestamp || new Date().toISOString(),
      });

      // 소리 재생
      if (soundEnabled) {
        const severity = (message.severity || 'info').toLowerCase();
        alertSound.play(severity as 'critical' | 'warning' | 'info');
      }
    });

    // 이상감지 메시지 핸들러
    const unsubAnomaly = wsService.onMessage('anomaly', (message: any) => {
      addLog('alert', `🔍 이상감지: [${message.severity}] Score=${message.score}`);

      addAnomaly({
        anomaly_id: `anomaly_${Date.now()}`,
        equipment_id: message.equipment_id || 'UNKNOWN',
        severity: message.severity || 'WARNING',
        anomaly_score: message.score || 0,
        detected_at: message.timestamp || new Date().toISOString(),
        message: message.message,
      });

      if (soundEnabled) {
        alertSound.play('warning');
      }
    });

    // 연결 시작
    wsService.connect();

    return () => {
      unsubConnect();
      unsubDisconnect();
      unsubAlert();
      unsubAnomaly();
    };
  }, [addLog, addAlarm, addAnomaly, soundEnabled]);

  // 오디오 컨텍스트 초기화 (사용자 인터랙션 필요)
  const initializeAudio = async () => {
    const success = await alertSound.initialize();
    if (success) {
      addLog('success', '오디오 초기화 완료');
      alertSound.playTest();
    } else {
      addLog('error', '오디오 초기화 실패');
    }
  };

  // 테스트 알람 발송
  const sendTestAlarm = async (severity: string) => {
    try {
      addLog('info', `테스트 알람 발송 중... (${severity})`);

      const params = new URLSearchParams({
        alert_type: 'test',
        equipment_id: 'TEST-001',
        severity,
        message: `[테스트] ${severity} 알람 - ${format(new Date(), 'HH:mm:ss')}`,
      });

      const response = await fetch(
        `${apiUrl}/api/realtime/broadcast/alert?${params}`,
        { method: 'POST' }
      );

      if (response.ok) {
        const data = await response.json();
        addLog('success', `알람 발송 완료 (수신자: ${data.recipients}명)`);
      } else {
        addLog('error', `알람 발송 실패: ${response.status}`);
      }
    } catch (e) {
      addLog('error', `알람 발송 오류: ${e}`);
    }
  };

  // 테스트 이상감지 발송
  const sendTestAnomaly = async () => {
    try {
      addLog('info', '테스트 이상감지 발송 중...');

      const params = new URLSearchParams({
        equipment_id: 'TEST-001',
        anomaly_type: 'test',
        severity: 'WARNING',
        score: (0.7 + Math.random() * 0.3).toFixed(3),
        message: `[테스트] 이상 패턴 감지 - ${format(new Date(), 'HH:mm:ss')}`,
      });

      const response = await fetch(
        `${apiUrl}/api/realtime/broadcast/anomaly?${params}`,
        { method: 'POST' }
      );

      if (response.ok) {
        addLog('success', '이상감지 발송 완료');
      } else {
        addLog('error', `이상감지 발송 실패: ${response.status}`);
      }
    } catch (e) {
      addLog('error', `이상감지 발송 오류: ${e}`);
    }
  };

  // 로그 색상 결정
  const getLogColor = (type: LogEntry['type']) => {
    switch (type) {
      case 'success':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
      case 'alert':
        return 'text-amber-400';
      default:
        return 'text-slate-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Alert System Test</h1>
          <p className="text-slate-400 text-sm mt-1">
            E2E 알람 시스템 테스트 페이지
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {/* 연결 상태 */}
          <div
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-full ${
              isConnected ? 'bg-green-500/20' : 'bg-red-500/20'
            }`}
          >
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
              }`}
            />
            <span
              className={`text-sm ${
                isConnected ? 'text-green-400' : 'text-red-400'
              }`}
            >
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 설정 */}
        <div className="card p-4">
          <h3 className="font-medium text-white mb-4">설정</h3>

          <div className="space-y-4">
            {/* API URL */}
            <div>
              <label className="block text-sm text-slate-400 mb-1">
                API URL
              </label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white text-sm"
              />
            </div>

            {/* 사운드 토글 */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-300">알람 소리</span>
              <button
                onClick={() => setSoundEnabled(!soundEnabled)}
                className={`px-3 py-1 rounded text-sm ${
                  soundEnabled
                    ? 'bg-blue-500 text-white'
                    : 'bg-slate-600 text-slate-300'
                }`}
              >
                {soundEnabled ? '🔊 ON' : '🔇 OFF'}
              </button>
            </div>

            {/* 오디오 초기화 */}
            <button
              onClick={initializeAudio}
              className="w-full px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded text-sm"
            >
              🔈 오디오 초기화 (클릭 필요)
            </button>
          </div>
        </div>

        {/* 테스트 버튼 */}
        <div className="card p-4">
          <h3 className="font-medium text-white mb-4">테스트 알람 발송</h3>

          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => sendTestAlarm('CRITICAL')}
              className="px-4 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/50 rounded font-medium"
            >
              🔴 CRITICAL
            </button>
            <button
              onClick={() => sendTestAlarm('WARNING')}
              className="px-4 py-3 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 border border-amber-500/50 rounded font-medium"
            >
              🟡 WARNING
            </button>
            <button
              onClick={() => sendTestAlarm('INFO')}
              className="px-4 py-3 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 border border-blue-500/50 rounded font-medium"
            >
              🔵 INFO
            </button>
            <button
              onClick={sendTestAnomaly}
              className="px-4 py-3 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/50 rounded font-medium"
            >
              🔍 Anomaly
            </button>
          </div>

          {/* 소리 테스트 */}
          <div className="mt-4 pt-4 border-t border-slate-700">
            <p className="text-sm text-slate-400 mb-2">소리 테스트</p>
            <div className="flex space-x-2">
              <button
                onClick={() => alertSound.playCritical()}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded text-xs"
              >
                Critical Sound
              </button>
              <button
                onClick={() => alertSound.playWarning()}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded text-xs"
              >
                Warning Sound
              </button>
              <button
                onClick={() => alertSound.playInfo()}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded text-xs"
              >
                Info Sound
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <p className="text-sm text-slate-400">Active Alarms</p>
          <p className="text-2xl font-bold text-white mt-1">
            {activeAlarms.length}
          </p>
        </div>
        <div className="card p-4">
          <p className="text-sm text-slate-400">Anomalies</p>
          <p className="text-2xl font-bold text-white mt-1">
            {recentAnomalies.length}
          </p>
        </div>
        <div className="card p-4">
          <p className="text-sm text-slate-400">Critical</p>
          <p className="text-2xl font-bold text-red-400 mt-1">
            {activeAlarms.filter((a) => a.severity === 'CRITICAL').length}
          </p>
        </div>
        <div className="card p-4">
          <p className="text-sm text-slate-400">Log Entries</p>
          <p className="text-2xl font-bold text-white mt-1">{logs.length}</p>
        </div>
      </div>

      {/* Log Console */}
      <div className="card">
        <div className="card-header flex items-center justify-between">
          <h3 className="font-medium text-white">Console Log</h3>
          <button
            onClick={() => setLogs([])}
            className="text-sm text-slate-400 hover:text-white"
          >
            Clear
          </button>
        </div>
        <div className="h-80 overflow-y-auto bg-slate-900 p-4 font-mono text-xs">
          {logs.length === 0 ? (
            <p className="text-slate-500">로그가 없습니다...</p>
          ) : (
            logs.map((log) => (
              <div key={log.id} className="py-0.5">
                <span className="text-slate-500">[{log.time}]</span>{' '}
                <span className={getLogColor(log.type)}>{log.message}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Recent Alarms */}
      {activeAlarms.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Recent Alarms</h3>
          </div>
          <div className="divide-y divide-slate-700">
            {activeAlarms.slice(0, 5).map((alarm) => (
              <div
                key={alarm.alarm_id}
                className={`px-4 py-3 ${
                  alarm.severity === 'CRITICAL'
                    ? 'border-l-4 border-red-500'
                    : alarm.severity === 'WARNING'
                    ? 'border-l-4 border-amber-500'
                    : 'border-l-4 border-blue-500'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-medium ${
                      alarm.severity === 'CRITICAL'
                        ? 'bg-red-500/20 text-red-400'
                        : alarm.severity === 'WARNING'
                        ? 'bg-amber-500/20 text-amber-400'
                        : 'bg-blue-500/20 text-blue-400'
                    }`}
                  >
                    {alarm.severity}
                  </span>
                  <span className="text-sm text-slate-400">
                    {alarm.equipment_id}
                  </span>
                </div>
                <p className="text-sm text-white mt-1">{alarm.message}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

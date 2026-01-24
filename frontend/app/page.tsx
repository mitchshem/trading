'use client'

import React, { useEffect, useRef, useState } from 'react'
import { createChart, IChartApi, ISeriesApi } from 'lightweight-charts'

// Frontend ↔ Backend Communication
// Frontend calls backend REST API endpoints and WebSocket for real-time updates
// Backend runs on http://localhost:8000 (configured in backend/main.py)
// CORS is configured to allow requests from http://localhost:3000 (frontend)
const API_BASE = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws'

interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface Signal {
  id: number
  timestamp: string
  symbol: string
  signal: 'BUY' | 'EXIT' | 'HOLD'
  price: number
  reason: string
}

interface Account {
  equity: number
  daily_pnl: number
  daily_realized_pnl: number
  open_positions: Array<{
    symbol: string
    shares: number
    entry_price: number
    current_price: number
    unrealized_pnl: number
    stop_price: number
  }>
  trade_blocked: boolean
  max_daily_loss: number
}

interface Trade {
  id: number
  symbol: string
  entry_time: string
  entry_price: number
  exit_time: string | null
  exit_price: number | null
  shares: number
  pnl: number | null
  reason: string | null
}

interface Metrics {
  metadata: {
    start_date: string | null
    end_date: string | null
    trade_count: number
    equity_start: number
    equity_end: number
  }
  core_metrics: {
    total_return_pct: number
    net_pnl: number
    win_rate: number
    loss_rate: number
    profit_factor: number | null
    expectancy_per_trade: number
    average_win: number
    average_loss: number
  }
  risk_metrics: {
    max_drawdown_absolute: number
    max_drawdown_pct: number
    max_consecutive_losses: number
    max_consecutive_wins: number
    exposure_pct: number
  }
  time_metrics: {
    trades_per_day: number
    avg_trade_duration_hours: number
    profitable_days_pct: number
  }
  risk_adjusted: {
    sharpe_proxy: number | null
  }
}

interface EquityPoint {
  timestamp: string
  equity: number
}

export default function Home() {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  
  const [symbols, setSymbols] = useState<string[]>([])
  const [selectedSymbol, setSelectedSymbol] = useState<string>('')
  const [isConnected, setIsConnected] = useState(false)
  const [signals, setSignals] = useState<Signal[]>([])
  const [account, setAccount] = useState<Account | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([])
  const markersRef = useRef<any[]>([])
  const updateMarkersRef = useRef<((signals: Signal[]) => void) | null>(null)
  const positionLinesRef = useRef<Array<{ time: number; price: number; type: 'entry' | 'exit' }>>([])
  const equityChartContainerRef = useRef<HTMLDivElement>(null)
  const equityChartRef = useRef<IChartApi | null>(null)
  const equitySeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  
  // Replay state
  const [replaySymbol, setReplaySymbol] = useState<string>('AAPL') // Default symbol
  const [replayStartDate, setReplayStartDate] = useState<string>('')
  const [replayEndDate, setReplayEndDate] = useState<string>('')
  const [replayStatus, setReplayStatus] = useState<any>(null)
  const [replayResults, setReplayResults] = useState<any>(null)
  const [isReplayRunning, setIsReplayRunning] = useState(false)
  const [backendError, setBackendError] = useState<string | null>(null)

  // Calculate default date range (last 6 months)
  const getDefaultDateRange = () => {
    const endDate = new Date()
    const startDate = new Date()
    startDate.setMonth(startDate.getMonth() - 6)
    
    return {
      start: startDate.toISOString().split('T')[0],
      end: endDate.toISOString().split('T')[0]
    }
  }

  // Fetch symbols on mount and set defaults
  useEffect(() => {
    fetch(`${API_BASE}/symbols`)
      .then(res => {
        if (!res.ok) {
          throw new Error('Backend not responding')
        }
        return res.json()
      })
      .then(data => {
        setBackendError(null)
        setSymbols(data.symbols || [])
        if (data.symbols && data.symbols.length > 0) {
          setSelectedSymbol(data.symbols[0])
          // Set default replay symbol if not already set
          if (!replaySymbol || !data.symbols.includes(replaySymbol)) {
            setReplaySymbol(data.symbols[0])
          }
        }
        // Set default date range
        const defaultDates = getDefaultDateRange()
        setReplayStartDate(defaultDates.start)
        setReplayEndDate(defaultDates.end)
      })
      .catch(err => {
        console.error('Failed to fetch symbols:', err)
        setBackendError('Backend not connected. Please ensure backend is running on http://localhost:8000')
        // Still set default dates even if backend fails
        const defaultDates = getDefaultDateRange()
        setReplayStartDate(defaultDates.start)
        setReplayEndDate(defaultDates.end)
      })
  }, [])

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600,
      layout: {
        background: { color: '#1a1a1a' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#2a2a2a' },
        horzLines: { color: '#2a2a2a' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries

    // Store price line references
    const priceLinesRef: { entry?: any; exit?: any } = {}

    // Function to update markers
    const updateMarkers = (signals: Signal[]) => {
      if (!candlestickSeriesRef.current) return
      
      // Clear existing markers
      candlestickSeriesRef.current.setMarkers([])
      markersRef.current = []
      
      // Add markers for BUY and EXIT signals
      const newMarkers = signals
        .filter(s => s.signal === 'BUY' || s.signal === 'EXIT')
        .map(s => {
          const timestamp = new Date(s.timestamp).getTime() / 1000 // Convert to Unix seconds
          return {
            time: timestamp as any,
            position: s.signal === 'BUY' ? 'belowBar' as const : 'aboveBar' as const,
            color: s.signal === 'BUY' ? '#26a69a' : '#ef5350',
            shape: s.signal === 'BUY' ? 'arrowUp' as const : 'arrowDown' as const,
            text: s.signal,
            size: 1,
          }
        })
      
      if (newMarkers.length > 0) {
        candlestickSeriesRef.current.setMarkers(newMarkers)
        markersRef.current = newMarkers
      }
    }

    // Function to update position lines
    const updatePositionLines = (trades: Trade[], symbol: string) => {
      if (!candlestickSeriesRef.current) return
      
      // Remove existing price lines
      if (priceLinesRef.entry) {
        candlestickSeriesRef.current.removePriceLine(priceLinesRef.entry)
        priceLinesRef.entry = undefined
      }
      if (priceLinesRef.exit) {
        candlestickSeriesRef.current.removePriceLine(priceLinesRef.exit)
        priceLinesRef.exit = undefined
      }
      
      // Find open or most recent trade for this symbol
      const symbolTrades = trades.filter(t => t.symbol === symbol)
      if (symbolTrades.length === 0) {
        return
      }
      
      // Find open trade first, otherwise use most recent closed trade
      const openTrade = symbolTrades.find(t => !t.exit_time)
      const trade = openTrade || symbolTrades[symbolTrades.length - 1]
      
      if (trade) {
        // Add entry line
        priceLinesRef.entry = candlestickSeriesRef.current.createPriceLine({
          price: trade.entry_price,
          color: '#26a69a',
          lineWidth: 2,
          lineStyle: 0, // Solid
          axisLabelVisible: true,
          title: `Entry: $${trade.entry_price.toFixed(2)}`,
        })
        
        // Add exit line if trade is closed
        if (trade.exit_price) {
          priceLinesRef.exit = candlestickSeriesRef.current.createPriceLine({
            price: trade.exit_price,
            color: '#ef5350',
            lineWidth: 2,
            lineStyle: 0, // Solid
            axisLabelVisible: true,
            title: `Exit: $${trade.exit_price.toFixed(2)}`,
          })
        }
      }
    }

    // Store functions in refs
    updateMarkersRef.current = updateMarkers
    ;(chartRef.current as any).updatePositionLines = updatePositionLines

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  // Fetch account info
  // FIX 3: STATE UPDATE CONSISTENCY
  // Polling: Account state is polled every 5 seconds as primary source of truth.
  // WebSocket: Also refreshes account on trade execution (see WebSocket handler below).
  // Rationale: Account state changes on every candle close (equity updates), so polling
  // ensures UI stays current even if WebSocket misses updates.
  useEffect(() => {
    const fetchAccount = () => {
      fetch(`${API_BASE}/account`)
        .then(res => res.json())
        .then(data => setAccount(data))
        .catch(err => console.error('Failed to fetch account:', err))
    }
    
    fetchAccount()
    const interval = setInterval(fetchAccount, 5000) // Refresh every 5 seconds
    
    return () => clearInterval(interval)
  }, [])

  // Fetch trades
  // FIX 3: STATE UPDATE CONSISTENCY
  // Polling: Trades are polled every 5 seconds as primary source of truth.
  // WebSocket: Also refreshes trades on trade execution (see WebSocket handler below).
  // Rationale: Trades may be executed outside WebSocket flow (stop-loss, kill switch),
  // so polling ensures all trades are captured.
  useEffect(() => {
    const fetchTrades = () => {
      fetch(`${API_BASE}/trades?limit=100`)
        .then(res => res.json())
        .then(data => {
          if (data.trades) {
            setTrades(data.trades)
            // Update position lines on chart
            if (selectedSymbol && chartRef.current && (chartRef.current as any).updatePositionLines) {
              ;(chartRef.current as any).updatePositionLines(data.trades, selectedSymbol)
            }
          }
        })
        .catch(err => console.error('Failed to fetch trades:', err))
    }
    
    fetchTrades()
    const interval = setInterval(fetchTrades, 5000) // Refresh every 5 seconds
    
    return () => clearInterval(interval)
  }, [selectedSymbol])

  // Fetch metrics
  // FIX 3: STATE UPDATE CONSISTENCY
  // Polling: Metrics are polled every 10 seconds as primary source of truth.
  // Rationale: Metrics are computed from persisted data and change less frequently
  // than account/trades, so longer polling interval is sufficient.
  // FIX 2: EXPLICIT REPLAY ISOLATION - This fetches live trading metrics only.
  // Replay metrics are displayed separately in replay panel.
  useEffect(() => {
    const fetchMetrics = () => {
      fetch(`${API_BASE}/metrics`)
        .then(res => res.json())
        .then(data => setMetrics(data))
        .catch(err => console.error('Failed to fetch metrics:', err))
    }
    
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 10000) // Refresh every 10 seconds
    
    return () => clearInterval(interval)
  }, [])

  // Fetch equity curve
  // FIX 3: STATE UPDATE CONSISTENCY
  // Polling: Equity curve is polled every 10 seconds as primary source of truth.
  // Rationale: Equity updates on every candle close, but chart doesn't need
  // real-time updates, so 10-second interval is sufficient.
  // FIX 2: EXPLICIT REPLAY ISOLATION - This fetches live trading equity curve only.
  // Replay equity curve is displayed separately in replay panel.
  useEffect(() => {
    const fetchEquityCurve = () => {
      fetch(`${API_BASE}/equity-curve?limit=1000`)
        .then(res => res.json())
        .then(data => {
          if (data.equity_curve) {
            setEquityCurve(data.equity_curve)
          }
        })
        .catch(err => console.error('Failed to fetch equity curve:', err))
    }
    
    fetchEquityCurve()
    const interval = setInterval(fetchEquityCurve, 10000)
    
    return () => clearInterval(interval)
  }, [])

  // Initialize equity curve chart
  useEffect(() => {
    if (!equityChartContainerRef.current) return

    const chart = createChart(equityChartContainerRef.current, {
      width: equityChartContainerRef.current.clientWidth,
      height: 300,
      layout: {
        background: { color: '#1a1a1a' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#2a2a2a' },
        horzLines: { color: '#2a2a2a' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const lineSeries = chart.addLineSeries({
      color: '#26a69a',
      lineWidth: 2,
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    })

    equityChartRef.current = chart
    equitySeriesRef.current = lineSeries

    // Handle resize
    const handleResize = () => {
      if (equityChartContainerRef.current && equityChartRef.current) {
        equityChartRef.current.applyOptions({
          width: equityChartContainerRef.current.clientWidth,
        })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  // Update equity curve chart when data changes
  useEffect(() => {
    if (!equitySeriesRef.current || equityCurve.length === 0) return

    const equityData = equityCurve.map((point: EquityPoint) => ({
      time: Math.floor(new Date(point.timestamp).getTime() / 1000) as any,
      value: point.equity,
    }))
    
    equitySeriesRef.current.setData(equityData)
  }, [equityCurve])

  // Fetch signals when symbol changes
  useEffect(() => {
    if (!selectedSymbol) return

    fetch(`${API_BASE}/signals?symbol=${selectedSymbol}&limit=100`)
      .then(res => res.json())
      .then(data => {
        if (data.signals) {
          setSignals(data.signals)
          // Update markers on chart
          if (updateMarkersRef.current) {
            updateMarkersRef.current(data.signals)
          }
        }
      })
      .catch(err => console.error('Failed to fetch signals:', err))
  }, [selectedSymbol])

  // Fetch candles and setup WebSocket when symbol changes
  useEffect(() => {
    if (!selectedSymbol || !candlestickSeriesRef.current) return

    // Close existing WebSocket
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
      setIsConnected(false)
    }

    // Fetch historical candles
    fetch(`${API_BASE}/candles?symbol=${selectedSymbol}&limit=500`)
      .then(res => res.json())
      .then(data => {
        if (data.candles && candlestickSeriesRef.current) {
          // Convert candles to format expected by lightweight-charts
          const formattedCandles = data.candles.map((c: Candle) => ({
            time: c.time as any,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          }))
          
          candlestickSeriesRef.current.setData(formattedCandles)
          
          // Markers will be updated when signals are fetched
          
          // Setup WebSocket for live updates
          const ws = new WebSocket(WS_URL)
          
          ws.onopen = () => {
            setIsConnected(true)
            ws.send(JSON.stringify({ symbol: selectedSymbol }))
          }
          
          ws.onmessage = (event) => {
            try {
            const data = JSON.parse(event.data)
            if (data.candle && candlestickSeriesRef.current) {
              const candle = data.candle
                // Lightweight Charts expects Unix timestamp in seconds
              candlestickSeriesRef.current.update({
                  time: candle.time as any, // Unix timestamp in seconds
                open: candle.open,
                high: candle.high,
                low: candle.low,
                close: candle.close,
              })
                
                // If a signal was generated, refresh signals list
                if (data.signal) {
                  fetch(`${API_BASE}/signals?symbol=${selectedSymbol}&limit=100`)
                    .then(res => res.json())
                    .then(signalData => {
                      if (signalData.signals) {
                        setSignals(signalData.signals)
                        // Update markers
                        if (updateMarkersRef.current) {
                          updateMarkersRef.current(signalData.signals)
                        }
                      }
                    })
                    .catch(err => console.error('Failed to refresh signals:', err))
                }
                
                // If a trade was executed, refresh account and trades
                if (data.trade) {
                  fetch(`${API_BASE}/account`)
                    .then(res => res.json())
                    .then(data => setAccount(data))
                    .catch(err => console.error('Failed to refresh account:', err))
                  
                  fetch(`${API_BASE}/trades?limit=100`)
                    .then(res => res.json())
                    .then(data => {
                      if (data.trades) {
                        setTrades(data.trades)
                      }
                    })
                    .catch(err => console.error('Failed to refresh trades:', err))
                }
                
                // Handle kill switch
                if (data.kill_switch) {
                  alert('KILL SWITCH TRIGGERED: Daily loss limit breached. All positions closed.')
                  fetch(`${API_BASE}/account`)
                    .then(res => res.json())
                    .then(data => setAccount(data))
                    .catch(err => console.error('Failed to refresh account:', err))
                }
              }
            } catch (error) {
              console.error('Error processing WebSocket message:', error)
            }
          }
          
          ws.onerror = (error) => {
            console.error('WebSocket error:', error)
            setIsConnected(false)
          }
          
          ws.onclose = () => {
            setIsConnected(false)
          }
          
          wsRef.current = ws
        }
      })
      .catch(err => console.error('Failed to fetch candles:', err))

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
        setIsConnected(false)
      }
    }
  }, [selectedSymbol])

  return (
    <main style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '20px' }}>
        <label htmlFor="symbol-select" style={{ fontSize: '16px', fontWeight: '500' }}>
          Symbol:
        </label>
        <select
          id="symbol-select"
          value={selectedSymbol}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSelectedSymbol(e.target.value)}
          style={{
            padding: '8px 12px',
            fontSize: '16px',
            backgroundColor: '#2a2a2a',
            color: '#ffffff',
            border: '1px solid #3a3a3a',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          {symbols.map((symbol: string) => (
            <option key={symbol} value={symbol}>
              {symbol}
            </option>
          ))}
        </select>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '8px',
          fontSize: '14px',
          color: isConnected ? '#26a69a' : '#ef5350'
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: isConnected ? '#26a69a' : '#ef5350'
          }} />
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
      
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: '600px',
          border: '1px solid #3a3a3a',
          borderRadius: '4px',
          marginBottom: '20px',
        }}
      />
      
      {/* Account Panel */}
      <div style={{
        backgroundColor: '#2a2a2a',
        border: '1px solid #3a3a3a',
        borderRadius: '4px',
        padding: '20px',
        marginBottom: '20px',
      }}>
        <h2 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: '600' }}>
          Account
        </h2>
        {account ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '14px', marginBottom: '4px' }}>Equity</div>
              <div style={{ color: '#ffffff', fontSize: '20px', fontWeight: '600' }}>
                ${account.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '14px', marginBottom: '4px' }}>Daily P&L</div>
              <div style={{ 
                color: account.daily_pnl >= 0 ? '#26a69a' : '#ef5350', 
                fontSize: '20px', 
                fontWeight: '600' 
              }}>
                ${account.daily_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '14px', marginBottom: '4px' }}>Max Daily Loss</div>
              <div style={{ color: '#ffffff', fontSize: '20px', fontWeight: '600' }}>
                ${account.max_daily_loss.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '14px', marginBottom: '4px' }}>Status</div>
              <div style={{ 
                color: account.trade_blocked ? '#ef5350' : '#26a69a', 
                fontSize: '16px', 
                fontWeight: '600' 
              }}>
                {account.trade_blocked ? 'TRADING BLOCKED' : 'ACTIVE'}
              </div>
            </div>
          </div>
        ) : (
          <p style={{ color: '#9ca3af', fontSize: '14px' }}>Loading account info...</p>
        )}
        
        {account && account.open_positions.length > 0 && (
          <div style={{ marginTop: '20px' }}>
            <h3 style={{ marginBottom: '12px', fontSize: '16px', fontWeight: '600', color: '#d1d5db' }}>
              Open Positions
            </h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #3a3a3a' }}>
                    <th style={{ textAlign: 'left', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Symbol</th>
                    <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Shares</th>
                    <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Entry</th>
                    <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Current</th>
                    <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>P&L</th>
                    <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Stop</th>
                  </tr>
                </thead>
                <tbody>
                  {account.open_positions.map((pos: Account['open_positions'][0], idx: number) => (
                    <tr key={idx} style={{ borderBottom: '1px solid #3a3a3a' }}>
                      <td style={{ padding: '8px', color: '#d1d5db', fontWeight: '600' }}>{pos.symbol}</td>
                      <td style={{ padding: '8px', textAlign: 'right', color: '#d1d5db' }}>{pos.shares}</td>
                      <td style={{ padding: '8px', textAlign: 'right', color: '#d1d5db' }}>${pos.entry_price.toFixed(2)}</td>
                      <td style={{ padding: '8px', textAlign: 'right', color: '#d1d5db' }}>${pos.current_price.toFixed(2)}</td>
                      <td style={{ 
                        padding: '8px', 
                        textAlign: 'right', 
                        color: pos.unrealized_pnl >= 0 ? '#26a69a' : '#ef5350',
                        fontWeight: '600'
                      }}>
                        ${pos.unrealized_pnl.toFixed(2)}
                      </td>
                      <td style={{ padding: '8px', textAlign: 'right', color: '#9ca3af' }}>${pos.stop_price.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
      
      {/* Trades Table */}
      <div style={{
        backgroundColor: '#2a2a2a',
        border: '1px solid #3a3a3a',
        borderRadius: '4px',
        padding: '20px',
        marginBottom: '20px',
      }}>
        <h2 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: '600' }}>
          Trades
        </h2>
        {trades.length === 0 ? (
          <p style={{ color: '#9ca3af', fontSize: '14px' }}>
            No trades executed yet.
          </p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #3a3a3a' }}>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Symbol</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Entry Time</th>
                  <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Entry Price</th>
                  <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Shares</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Exit Time</th>
                  <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Exit Price</th>
                  <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>P&L</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Reason</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade: Trade) => (
                  <tr key={trade.id} style={{ borderBottom: '1px solid #3a3a3a' }}>
                    <td style={{ padding: '8px', color: '#d1d5db', fontWeight: '600' }}>{trade.symbol}</td>
                    <td style={{ padding: '8px', color: '#d1d5db' }}>
                      {/* FIX 1: CANONICAL TIME HANDLING - Backend sends UTC, convert to local for display */}
                      {new Date(trade.entry_time).toLocaleString()}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'right', color: '#d1d5db' }}>
                      ${trade.entry_price.toFixed(2)}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'right', color: '#d1d5db' }}>
                      {trade.shares}
                    </td>
                    <td style={{ padding: '8px', color: trade.exit_time ? '#d1d5db' : '#6b7280' }}>
                      {/* FIX 1: CANONICAL TIME HANDLING - Backend sends UTC, convert to local for display */}
                      {trade.exit_time ? new Date(trade.exit_time).toLocaleString() : 'Open'}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'right', color: trade.exit_price ? '#d1d5db' : '#6b7280' }}>
                      {trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : '-'}
                    </td>
                    <td style={{ 
                      padding: '8px', 
                      textAlign: 'right',
                      color: trade.pnl === null ? '#6b7280' : (trade.pnl >= 0 ? '#26a69a' : '#ef5350'),
                      fontWeight: trade.pnl !== null ? '600' : 'normal'
                    }}>
                      {trade.pnl !== null ? `$${trade.pnl.toFixed(2)}` : '-'}
                    </td>
                    <td style={{ padding: '8px', color: '#9ca3af', fontSize: '13px' }}>
                      {trade.reason || '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      
      {/* Performance Panel */}
      <div style={{
        backgroundColor: '#2a2a2a',
        border: '1px solid #3a3a3a',
        borderRadius: '4px',
        padding: '20px',
        marginBottom: '20px',
      }}>
        <h2 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: '600' }}>
          Performance Metrics
        </h2>
        
        {metrics ? (
          <>
            {/* Equity Curve Chart */}
            <div style={{ marginBottom: '30px' }}>
              <h3 style={{ marginBottom: '12px', fontSize: '16px', fontWeight: '600', color: '#d1d5db' }}>
                Equity Curve
              </h3>
              <div
                ref={equityChartContainerRef}
                style={{
                  width: '100%',
                  height: '300px',
                  border: '1px solid #3a3a3a',
                  borderRadius: '4px',
                }}
              />
            </div>
            
            {/* Returns Section */}
            <div style={{ marginBottom: '30px' }}>
              <h3 style={{ marginBottom: '12px', fontSize: '16px', fontWeight: '600', color: '#d1d5db' }}>
                Returns
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Total Return</div>
                  <div style={{ 
                    color: metrics.core_metrics.total_return_pct >= 0 ? '#26a69a' : '#ef5350', 
                    fontSize: '18px', 
                    fontWeight: '600' 
                  }}>
                    {metrics.core_metrics.total_return_pct.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Net P&L</div>
                  <div style={{ 
                    color: metrics.core_metrics.net_pnl >= 0 ? '#26a69a' : '#ef5350', 
                    fontSize: '18px', 
                    fontWeight: '600' 
                  }}>
                    ${metrics.core_metrics.net_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Win Rate</div>
                  <div style={{ color: '#ffffff', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.core_metrics.win_rate.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Profit Factor</div>
                  <div style={{ color: '#ffffff', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.core_metrics.profit_factor !== null ? metrics.core_metrics.profit_factor.toFixed(2) : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Expectancy</div>
                  <div style={{ 
                    color: metrics.core_metrics.expectancy_per_trade >= 0 ? '#26a69a' : '#ef5350', 
                    fontSize: '18px', 
                    fontWeight: '600' 
                  }}>
                    ${metrics.core_metrics.expectancy_per_trade.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Avg Win</div>
                  <div style={{ color: '#26a69a', fontSize: '18px', fontWeight: '600' }}>
                    ${metrics.core_metrics.average_win.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Avg Loss</div>
                  <div style={{ color: '#ef5350', fontSize: '18px', fontWeight: '600' }}>
                    ${metrics.core_metrics.average_loss.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Risk Section */}
            <div style={{ marginBottom: '30px' }}>
              <h3 style={{ marginBottom: '12px', fontSize: '16px', fontWeight: '600', color: '#d1d5db' }}>
                Risk
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Max Drawdown</div>
                  <div style={{ color: '#ef5350', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.risk_metrics.max_drawdown_pct.toFixed(2)}%
                  </div>
                  <div style={{ color: '#6b7280', fontSize: '12px', marginTop: '2px' }}>
                    ${metrics.risk_metrics.max_drawdown_absolute.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Max Consecutive Losses</div>
                  <div style={{ color: '#ef5350', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.risk_metrics.max_consecutive_losses}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Max Consecutive Wins</div>
                  <div style={{ color: '#26a69a', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.risk_metrics.max_consecutive_wins}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Exposure</div>
                  <div style={{ color: '#ffffff', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.risk_metrics.exposure_pct.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Sharpe Proxy</div>
                  <div style={{ color: '#ffffff', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.risk_adjusted.sharpe_proxy !== null ? metrics.risk_adjusted.sharpe_proxy.toFixed(2) : 'N/A'}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Consistency Section */}
            <div>
              <h3 style={{ marginBottom: '12px', fontSize: '16px', fontWeight: '600', color: '#d1d5db' }}>
                Consistency
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Trades per Day</div>
                  <div style={{ color: '#ffffff', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.time_metrics.trades_per_day.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Avg Trade Duration</div>
                  <div style={{ color: '#ffffff', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.time_metrics.avg_trade_duration_hours.toFixed(1)}h
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Profitable Days</div>
                  <div style={{ color: '#ffffff', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.time_metrics.profitable_days_pct.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9ca3af', fontSize: '13px', marginBottom: '4px' }}>Total Trades</div>
                  <div style={{ color: '#ffffff', fontSize: '18px', fontWeight: '600' }}>
                    {metrics.metadata.trade_count}
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <p style={{ color: '#9ca3af', fontSize: '14px' }}>Loading metrics...</p>
        )}
      </div>
      
      {/* Replay Panel */}
      <div style={{
        backgroundColor: '#2a2a2a',
        border: '1px solid #3a3a3a',
        borderRadius: '4px',
        padding: '20px',
        marginBottom: '20px',
      }}>
        <h2 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: '600' }}>
          Historical Replay (Backtest)
        </h2>
        
        {backendError && (
          <div style={{
            padding: '12px',
            backgroundColor: '#ef5350',
            color: '#ffffff',
            borderRadius: '4px',
            marginBottom: '16px',
            fontSize: '14px'
          }}>
            ⚠️ {backendError}
          </div>
        )}
        
        <div style={{ 
          padding: '12px', 
          backgroundColor: '#1a1a1a', 
          borderRadius: '4px',
          marginBottom: '16px',
          fontSize: '13px',
          color: '#9ca3af'
        }}>
          <div style={{ marginBottom: '8px', fontWeight: '600', color: '#d1d5db' }}>Defaults:</div>
          <div>Symbol: <span style={{ color: '#26a69a' }}>{replaySymbol || 'Not set'}</span></div>
          <div>Date Range: <span style={{ color: '#26a69a' }}>{replayStartDate || 'Not set'}</span> → <span style={{ color: '#26a69a' }}>{replayEndDate || 'Not set'}</span></div>
        </div>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center', marginBottom: '16px' }}>
          <label htmlFor="replay-symbol-select" style={{ fontSize: '14px', fontWeight: '500' }}>
            Symbol:
          </label>
          <select
            id="replay-symbol-select"
            value={replaySymbol}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setReplaySymbol(e.target.value)}
            disabled={isReplayRunning}
            style={{
              padding: '6px 10px',
              fontSize: '14px',
              backgroundColor: '#1a1a1a',
              color: '#ffffff',
              border: '1px solid #3a3a3a',
              borderRadius: '4px',
              cursor: isReplayRunning ? 'not-allowed' : 'pointer',
            }}
          >
            <option value="">Select symbol</option>
            {symbols.map((symbol: string) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
          
          <label htmlFor="replay-start-date" style={{ fontSize: '14px', fontWeight: '500', marginLeft: '12px' }}>
            Start Date:
          </label>
          <input
            id="replay-start-date"
            type="date"
            value={replayStartDate}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setReplayStartDate(e.target.value)}
            disabled={isReplayRunning}
            style={{
              padding: '6px 10px',
              fontSize: '14px',
              backgroundColor: '#1a1a1a',
              color: '#ffffff',
              border: '1px solid #3a3a3a',
              borderRadius: '4px',
              cursor: isReplayRunning ? 'not-allowed' : 'pointer',
            }}
          />
          
          <label htmlFor="replay-end-date" style={{ fontSize: '14px', fontWeight: '500', marginLeft: '12px' }}>
            End Date:
          </label>
          <input
            id="replay-end-date"
            type="date"
            value={replayEndDate}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setReplayEndDate(e.target.value)}
            disabled={isReplayRunning}
            style={{
              padding: '6px 10px',
              fontSize: '14px',
              backgroundColor: '#1a1a1a',
              color: '#ffffff',
              border: '1px solid #3a3a3a',
              borderRadius: '4px',
              cursor: isReplayRunning ? 'not-allowed' : 'pointer',
            }}
          />
          
          <button
            onClick={async () => {
              if (!replaySymbol) {
                alert('Please select a symbol')
                return
              }
              
              if (!replayStartDate || !replayEndDate) {
                alert('Please select both start and end dates')
                return
              }
              
              setIsReplayRunning(true)
              setReplayStatus({ status: 'Running...' })
              setReplayResults(null)
              setBackendError(null)
              
              try {
                // Start replay with date range (will fetch from Yahoo Finance automatically)
                const replayRes = await fetch(`${API_BASE}/replay/start`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    symbol: replaySymbol,
                    start_date: replayStartDate,
                    end_date: replayEndDate
                  })
                })
                
                if (!replayRes.ok) {
                  const errorData = await replayRes.json().catch(() => ({ error: 'Backend error' }))
                  throw new Error(errorData.error || 'Replay failed')
                }
                
                const replayData = await replayRes.json()
                
                if (replayData.error) {
                  throw new Error(replayData.error)
                }
                
                // Fetch results
                const resultsRes = await fetch(`${API_BASE}/replay/results?replay_id=${replayData.replay_id}`)
                if (!resultsRes.ok) {
                  throw new Error('Failed to fetch replay results')
                }
                const resultsData = await resultsRes.json()
                
                setReplayResults(resultsData)
                setReplayStatus({
                  status: 'Completed',
                  replay_id: replayData.replay_id,
                  total_candles: replayData.total_candles,
                  final_equity: replayData.final_equity,
                  source: replayData.source || 'yahoo_finance',
                  determinism_status: replayData.determinism_status,
                  determinism_message: replayData.determinism_message,
                  determinism_mismatches: replayData.determinism_mismatches,
                  replay_fingerprint: replayData.replay_fingerprint
                })
                
                // Populate UI with replay results
                // 1. Update trades with replay trades
                if (resultsData.trades && resultsData.trades.length > 0) {
                  setTrades(resultsData.trades)
                }
                
                // 2. Update equity curve
                if (resultsData.equity_curve && resultsData.equity_curve.length > 0) {
                  setEquityCurve(resultsData.equity_curve)
                }
                
                // 3. Update metrics
                if (resultsData.metrics) {
                  setMetrics(resultsData.metrics)
                }
                
                // 4. Fetch signals for the replay symbol (replay signals are stored with replay_id)
                fetch(`${API_BASE}/signals?symbol=${replaySymbol}&limit=100`)
                  .then(res => res.json())
                  .then(data => {
                    if (data.signals) {
                      // Filter to only show signals from this replay if possible
                      setSignals(data.signals)
                      // Update markers on chart
                      if (updateMarkersRef.current) {
                        updateMarkersRef.current(data.signals)
                      }
                    }
                  })
                  .catch(err => console.error('Failed to fetch signals:', err))
                
                // 5. Update selected symbol to trigger chart refresh
                setSelectedSymbol(replaySymbol)
                
              } catch (error: any) {
                setBackendError(error.message || 'Replay failed')
                setReplayStatus({ status: 'Error', error: error.message })
                alert(`Replay failed: ${error.message}`)
              } finally {
                setIsReplayRunning(false)
              }
            }}
            disabled={isReplayRunning || !replaySymbol || !replayStartDate || !replayEndDate || !!backendError}
            style={{
              padding: '8px 16px',
              fontSize: '14px',
              fontWeight: '600',
              backgroundColor: isReplayRunning || !replaySymbol || !replayStartDate || !replayEndDate || !!backendError ? '#3a3a3a' : '#26a69a',
              color: '#ffffff',
              border: 'none',
              borderRadius: '4px',
              cursor: isReplayRunning || !replaySymbol || !replayStartDate || !replayEndDate || !!backendError ? 'not-allowed' : 'pointer',
            }}
          >
            {isReplayRunning ? 'Running...' : 'Run Replay'}
          </button>
        </div>
        
        {replayStatus && (
          <div style={{ 
            padding: '12px', 
            backgroundColor: '#1a1a1a', 
            borderRadius: '4px',
            marginBottom: '16px'
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', fontSize: '14px' }}>
              <div>
                <div style={{ color: '#9ca3af', marginBottom: '4px' }}>Status</div>
                <div style={{ 
                  color: replayStatus.status === 'Completed' ? '#26a69a' : replayStatus.status === 'Error' ? '#ef5350' : '#ffa726',
                  fontWeight: '600' 
                }}>
                  {replayStatus.status}
                </div>
              </div>
              {replayStatus.total_candles && (
                <div>
                  <div style={{ color: '#9ca3af', marginBottom: '4px' }}>Candles Processed</div>
                  <div style={{ color: '#ffffff', fontWeight: '600' }}>{replayStatus.total_candles.toLocaleString()}</div>
                </div>
              )}
              {replayStatus.final_equity && (
                <div>
                  <div style={{ color: '#9ca3af', marginBottom: '4px' }}>Final Equity</div>
                  <div style={{ color: '#ffffff', fontWeight: '600' }}>
                    ${replayStatus.final_equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                </div>
              )}
              {replayStatus.error && (
                <div>
                  <div style={{ color: '#9ca3af', marginBottom: '4px' }}>Error</div>
                  <div style={{ color: '#ef5350', fontWeight: '600', fontSize: '12px' }}>{replayStatus.error}</div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {replayResults && replayResults.metrics && (
          <div style={{ 
            padding: '12px', 
            backgroundColor: '#1a1a1a', 
            borderRadius: '4px',
            fontSize: '13px'
          }}>
            <div style={{ color: '#9ca3af', marginBottom: '8px' }}>Replay Results Summary</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px' }}>
              <div>
                <div style={{ color: '#6b7280' }}>Total Return</div>
                <div style={{ 
                  color: replayResults.metrics.core_metrics.total_return_pct >= 0 ? '#26a69a' : '#ef5350',
                  fontWeight: '600'
                }}>
                  {replayResults.metrics.core_metrics.total_return_pct.toFixed(2)}%
                </div>
              </div>
              <div>
                <div style={{ color: '#6b7280' }}>Net P&L</div>
                <div style={{ 
                  color: replayResults.metrics.core_metrics.net_pnl >= 0 ? '#26a69a' : '#ef5350',
                  fontWeight: '600'
                }}>
                  ${replayResults.metrics.core_metrics.net_pnl.toFixed(2)}
                </div>
              </div>
              <div>
                <div style={{ color: '#6b7280' }}>Win Rate</div>
                <div style={{ color: '#ffffff', fontWeight: '600' }}>
                  {replayResults.metrics.core_metrics.win_rate.toFixed(2)}%
                </div>
              </div>
              <div>
                <div style={{ color: '#6b7280' }}>Trades</div>
                <div style={{ color: '#ffffff', fontWeight: '600' }}>
                  {replayResults.metrics.metadata.trade_count}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Signals Panel */}
      <div style={{
        backgroundColor: '#2a2a2a',
        border: '1px solid #3a3a3a',
        borderRadius: '4px',
        padding: '20px',
      }}>
        <h2 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: '600' }}>
          Trading Signals
        </h2>
        {signals.length === 0 ? (
          <p style={{ color: '#9ca3af', fontSize: '14px' }}>
            No signals generated yet. Signals will appear here when strategy conditions are met.
          </p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #3a3a3a' }}>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Time</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Signal</th>
                  <th style={{ textAlign: 'right', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Price</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#9ca3af', fontWeight: '500' }}>Reason</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((signal) => (
                  <tr key={signal.id} style={{ borderBottom: '1px solid #3a3a3a' }}>
                    <td style={{ padding: '8px', color: '#d1d5db' }}>
                      {/* FIX 1: CANONICAL TIME HANDLING - Backend sends UTC, convert to local for display */}
                      {new Date(signal.timestamp).toLocaleString()}
                    </td>
                    <td style={{ padding: '8px' }}>
                      <span style={{
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        fontWeight: '600',
                        backgroundColor: signal.signal === 'BUY' ? '#26a69a' : signal.signal === 'EXIT' ? '#ef5350' : '#6b7280',
                        color: '#ffffff',
                      }}>
                        {signal.signal}
                      </span>
                    </td>
                    <td style={{ padding: '8px', textAlign: 'right', color: '#d1d5db' }}>
                      ${signal.price.toFixed(2)}
                    </td>
                    <td style={{ padding: '8px', color: '#9ca3af', fontSize: '13px' }}>
                      {signal.reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  )
}

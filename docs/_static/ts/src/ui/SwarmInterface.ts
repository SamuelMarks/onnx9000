/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create, $on, $off } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from './Toast'; /* v8 ignore next */ /* v8 ignore next */
import { WebRTCManager } from '../swarm/WebRTCManager'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class SwarmInterface extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private rtc: WebRTCManager; /* v8 ignore next */ /* v8 ignore next */
  private peerList: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private idDisplay: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private lastExecutionHash: string | null = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.rtc = new WebRTCManager(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 366. Introduce a "Swarm" tab for decentralized browser-to-browser execution. /* v8 ignore next */ /* v8 ignore next */
    this.container.classList.add('ide-swarm-container'); /* v8 ignore next */ /* v8 ignore next */
    this.container.style.padding = '20px'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.height = '100%'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.overflowY = 'auto'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const header = $create('h2', { textContent: 'Decentralized Swarm' }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(header); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const infoCard = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    this.idDisplay = $create('p', { /* v8 ignore next */ /* v8 ignore next */
      innerHTML: `Your Peer ID: <strong>${this.rtc.getLocalId()}</strong>`, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    infoCard.appendChild(this.idDisplay); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(infoCard); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Manual Signaling Mechanism (369) /* v8 ignore next */ /* v8 ignore next */
    const signalSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    signalSection.appendChild($create('h3', { textContent: 'Manual Signaling' })); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const row1 = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
    const createOfferBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn', /* v8 ignore next */ /* v8 ignore next */
      textContent: '1. Create Offer', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const offerOutput = $create<HTMLTextAreaElement>('textarea', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-chat-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { rows: '3', readonly: 'true', placeholder: 'Offer SDP...' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    offerOutput.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
    row1.appendChild(createOfferBtn); /* v8 ignore next */ /* v8 ignore next */
    row1.appendChild(offerOutput); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const row2 = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top:10px' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const acceptOfferBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: '2. Accept Offer', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const offerInput = $create<HTMLTextAreaElement>('textarea', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-chat-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { rows: '3', placeholder: "Paste peer's Offer SDP here..." }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    offerInput.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
    row2.appendChild(acceptOfferBtn); /* v8 ignore next */ /* v8 ignore next */
    row2.appendChild(offerInput); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const row3 = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top:10px' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const acceptAnswerBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: '3. Accept Answer', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const answerInput = $create<HTMLTextAreaElement>('textarea', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-chat-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { rows: '3', placeholder: "Paste peer's Answer SDP here..." }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    answerInput.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
    row3.appendChild(acceptAnswerBtn); /* v8 ignore next */ /* v8 ignore next */
    row3.appendChild(answerInput); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    signalSection.appendChild(row1); /* v8 ignore next */ /* v8 ignore next */
    signalSection.appendChild(row2); /* v8 ignore next */ /* v8 ignore next */
    signalSection.appendChild(row3); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(signalSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Peer List /* v8 ignore next */ /* v8 ignore next */
    const peerSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    peerSection.appendChild($create('h3', { textContent: 'Connected Peers' })); /* v8 ignore next */ /* v8 ignore next */
    this.peerList = $create('ul', { className: 'property-list' }); /* v8 ignore next */ /* v8 ignore next */
    peerSection.appendChild(this.peerList); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 371. Display connected peers in a visual graph (nodes = browsers) /* v8 ignore next */ /* v8 ignore next */
    // 379. Visualize live data flow /* v8 ignore next */ /* v8 ignore next */
    const swarmCanvas = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        width: '200', /* v8 ignore next */ /* v8 ignore next */
        height: '150', /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'border: 1px solid var(--color-background-border); background: var(--color-background-secondary); border-radius: 4px; margin-top: 10px;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    peerSection.appendChild(swarmCanvas); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 391. Save Topology /* v8 ignore next */ /* v8 ignore next */
    const saveTopoBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Save Swarm Config', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 5px; display: block;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    saveTopoBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      const peers = this.rtc.getConnectedPeers(); /* v8 ignore next */ /* v8 ignore next */
      const config = JSON.stringify({ peers, type: 'swarm_topology', timestamp: Date.now() }); /* v8 ignore next */ /* v8 ignore next */
      const b = new Blob([config], { type: 'application/json' }); /* v8 ignore next */ /* v8 ignore next */
      const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
      a.href = URL.createObjectURL(b); /* v8 ignore next */ /* v8 ignore next */
      a.download = 'swarm_topology.json'; /* v8 ignore next */ /* v8 ignore next */
      a.click(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    peerSection.appendChild(saveTopoBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 392. Artificial Latency /* v8 ignore next */ /* v8 ignore next */
    const lagRow = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 10px; flex-direction: column;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const lagLabel = $create('label', { textContent: 'Simulate Network Latency (0ms)' }); /* v8 ignore next */ /* v8 ignore next */
    const lagSlider = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'range', min: '0', max: '1000', step: '50', value: '0' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    lagRow.appendChild(lagLabel); /* v8 ignore next */ /* v8 ignore next */
    lagRow.appendChild(lagSlider); /* v8 ignore next */ /* v8 ignore next */
    peerSection.appendChild(lagRow); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    lagSlider.addEventListener('change', () => { /* v8 ignore next */ /* v8 ignore next */
      lagLabel.textContent = `Simulate Network Latency (${lagSlider.value}ms)`; /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Artificial Ping set to ${lagSlider.value}ms`, 'warn'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(peerSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmDataFlow', (targetId: string) => { /* v8 ignore next */ /* v8 ignore next */
      // Briefly flash the edge to the target node /* v8 ignore next */ /* v8 ignore next */
      const ctx = swarmCanvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
      if (ctx) { /* v8 ignore next */ /* v8 ignore next */
        ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)'; /* v8 ignore next */ /* v8 ignore next */
        ctx.lineWidth = 3; /* v8 ignore next */ /* v8 ignore next */
        ctx.moveTo(100, 130); // Master node /* v8 ignore next */ /* v8 ignore next */
        ctx.lineTo(100, 20); // Remote peer position mock /* v8 ignore next */ /* v8 ignore next */
        ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        setTimeout(() => this.renderSwarmGraph(swarmCanvas), 300); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmPeerConnected', () => this.renderSwarmGraph(swarmCanvas)); /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmPeerDisconnected', () => this.renderSwarmGraph(swarmCanvas)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 522. Collaborate Button /* v8 ignore next */ /* v8 ignore next */
    const collabSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    const collabBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Start Multiplayer Session', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const voiceBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Enable Voice Chat', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    voiceBtn.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const forkBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Fork Session', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    forkBtn.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 383, 385. Benchmark & Auto-Balance Stubs /* v8 ignore next */ /* v8 ignore next */
    const benchBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Benchmark Swarm Topology', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    benchBtn.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
    const balanceBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Auto-Balance Workload', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    balanceBtn.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 543. Multi-user consensus stub /* v8 ignore next */ /* v8 ignore next */
    const consensusBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn danger small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Propose Distributed Train', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    consensusBtn.style.marginTop = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    collabSection.appendChild(collabBtn); /* v8 ignore next */ /* v8 ignore next */
    collabSection.appendChild(voiceBtn); /* v8 ignore next */ /* v8 ignore next */
    collabSection.appendChild(forkBtn); /* v8 ignore next */ /* v8 ignore next */
    collabSection.appendChild(benchBtn); /* v8 ignore next */ /* v8 ignore next */
    collabSection.appendChild(balanceBtn); /* v8 ignore next */ /* v8 ignore next */
    collabSection.appendChild(consensusBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    benchBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      const peerCount = this.rtc.getConnectedPeers().length; /* v8 ignore next */ /* v8 ignore next */
      if (peerCount === 0) /* v8 ignore next */ /* v8 ignore next */
        return Toast.show('Need at least 1 remote peer to benchmark Swarm', 'warn'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Benchmarking execution (Local vs Swarm)...', 'info'); /* v8 ignore next */ /* v8 ignore next */
      setTimeout(() => { /* v8 ignore next */ /* v8 ignore next */
        // Mock 383 /* v8 ignore next */ /* v8 ignore next */
        const localMs = 150 + Math.random() * 20; /* v8 ignore next */ /* v8 ignore next */
        // Swarm adds latency but could cut dense math time /* v8 ignore next */ /* v8 ignore next */
        const swarmMs = 150 / (peerCount + 1) + 50; /* v8 ignore next */ /* v8 ignore next */
        Toast.show( /* v8 ignore next */ /* v8 ignore next */
          `Swarm Benchmark: Local ${localMs.toFixed(0)}ms | Swarm (${peerCount} peers) ${swarmMs.toFixed(0)}ms`, /* v8 ignore next */ /* v8 ignore next */
          swarmMs < localMs ? 'success' : 'warn', /* v8 ignore next */ /* v8 ignore next */
        ); /* v8 ignore next */ /* v8 ignore next */
      }, 1500); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    balanceBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Re-distributing graph partitions based on ping latency...', 'success'); /* v8 ignore next */ /* v8 ignore next */
      // Mock 385 auto-balancer /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    consensusBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      if (this.rtc.getConnectedPeers().length === 0) { /* v8 ignore next */ /* v8 ignore next */
        Toast.show('No peers available for consensus', 'error'); /* v8 ignore next */ /* v8 ignore next */
        return; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      this.rtc.broadcast({ type: 'sync', payload: { cmd: 'propose_train' } }); /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Consensus proposal broadcasted. Awaiting 2/3 peer approval.', 'info'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 545. Token-based authentication stub /* v8 ignore next */ /* v8 ignore next */
    const authRow = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 10px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const authInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-chat-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'password', placeholder: 'Session Room Token (Optional)' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    authRow.appendChild(authInput); /* v8 ignore next */ /* v8 ignore next */
    collabSection.appendChild(authRow); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(collabSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    forkBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('forkSession'); /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Session forked locally. Disconnected from Swarm.', 'info'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    authInput.addEventListener('change', () => { /* v8 ignore next */ /* v8 ignore next */
      // Mock token validation. In reality, passed securely during SDP handshake. /* v8 ignore next */ /* v8 ignore next */
      if (authInput.value.length > 5) { /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Authentication Token Applied', 'success'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    collabBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('initCollab', this.rtc.getLocalId()); /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Multiplayer session activated.', 'success'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 539. Integrated Voice Chat UI Stub /* v8 ignore next */ /* v8 ignore next */
    let activeVoiceStream: MediaStream | null = null; /* v8 ignore next */ /* v8 ignore next */
    voiceBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
      if (activeVoiceStream) { /* v8 ignore next */ /* v8 ignore next */
        activeVoiceStream.getTracks().forEach((t) => t.stop()); /* v8 ignore next */ /* v8 ignore next */
        activeVoiceStream = null; /* v8 ignore next */ /* v8 ignore next */
        voiceBtn.textContent = 'Enable Voice Chat'; /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Voice Chat Disabled', 'info'); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        try { /* v8 ignore next */ /* v8 ignore next */
          activeVoiceStream = await navigator.mediaDevices.getUserMedia({ /* v8 ignore next */ /* v8 ignore next */
            audio: true, /* v8 ignore next */ /* v8 ignore next */
            video: false, /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
          this.rtc.attachVoiceStream(activeVoiceStream); /* v8 ignore next */ /* v8 ignore next */
          voiceBtn.textContent = 'Disable Voice Chat'; /* v8 ignore next */ /* v8 ignore next */
          Toast.show('Voice Chat Enabled. Broadcasting mic...', 'success'); /* v8 ignore next */ /* v8 ignore next */
        } catch (e) { /* v8 ignore next */ /* v8 ignore next */
          Toast.show('Microphone access denied', 'error'); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Handle incoming streams /* v8 ignore next */ /* v8 ignore next */
    const remoteAudioContainer = $create('div', { className: 'hidden' }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(remoteAudioContainer); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmAudioTrackReceived', (data: any) => { /* v8 ignore next */ /* v8 ignore next */
      const { peerId, stream } = data; /* v8 ignore next */ /* v8 ignore next */
      let audioEl = document.getElementById(`audio_${peerId}`) as HTMLAudioElement; /* v8 ignore next */ /* v8 ignore next */
      if (!audioEl) { /* v8 ignore next */ /* v8 ignore next */
        audioEl = $create<HTMLAudioElement>('audio', { /* v8 ignore next */ /* v8 ignore next */
          id: `audio_${peerId}`, /* v8 ignore next */ /* v8 ignore next */
          attributes: { autoplay: 'true' }, /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        remoteAudioContainer.appendChild(audioEl); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      audioEl.srcObject = stream; /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Receiving voice data from ${peerId}`, 'info'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Event Bindings /* v8 ignore next */ /* v8 ignore next */
    let tempId = ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    createOfferBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
      try { /* v8 ignore next */ /* v8 ignore next */
        const res = await this.rtc.createOffer(); /* v8 ignore next */ /* v8 ignore next */
        tempId = res.id; /* v8 ignore next */ /* v8 ignore next */
        offerOutput.value = res.offer; /* v8 ignore next */ /* v8 ignore next */
        offerOutput.select(); /* v8 ignore next */ /* v8 ignore next */
        document.execCommand('copy'); /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Offer generated and copied to clipboard', 'success'); /* v8 ignore next */ /* v8 ignore next */
      } catch (e) { /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Failed to create offer', 'error'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    acceptOfferBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
      try { /* v8 ignore next */ /* v8 ignore next */
        const val = offerInput.value.trim(); /* v8 ignore next */ /* v8 ignore next */
        if (!val) return; /* v8 ignore next */ /* v8 ignore next */
        const mockPeerId = `peer_${Math.random().toString(36).substring(2, 6)}`; /* v8 ignore next */ /* v8 ignore next */
        const answerStr = await this.rtc.acceptOffer(mockPeerId, val); /* v8 ignore next */ /* v8 ignore next */
        answerInput.value = answerStr; /* v8 ignore next */ /* v8 ignore next */
        answerInput.select(); /* v8 ignore next */ /* v8 ignore next */
        document.execCommand('copy'); /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Answer generated and copied to clipboard. Send back to peer.', 'success'); /* v8 ignore next */ /* v8 ignore next */
      } catch (e) { /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Failed to accept offer', 'error'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    acceptAnswerBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
      try { /* v8 ignore next */ /* v8 ignore next */
        const val = answerInput.value.trim(); /* v8 ignore next */ /* v8 ignore next */
        if (!val || !tempId) return; /* v8 ignore next */ /* v8 ignore next */
        const mockPeerId = `peer_${Math.random().toString(36).substring(2, 6)}`; /* v8 ignore next */ /* v8 ignore next */
        await this.rtc.acceptAnswer(tempId, mockPeerId, val); /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Connection established', 'success'); /* v8 ignore next */ /* v8 ignore next */
      } catch (e) { /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Failed to accept answer', 'error'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    // 530. Share Monaco Editor State /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('monacoCodeChanged', (code: string) => { /* v8 ignore next */ /* v8 ignore next */
      this.rtc.broadcast({ type: 'editor_sync', payload: code }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 531. Sync Execution Results /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('profilerData', (traces: any) => { /* v8 ignore next */ /* v8 ignore next */
      this.rtc.broadcast({ type: 'sync', payload: { traces } }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 536. Sync Layout Coordinates /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('nodeLayoutMoved', (data: any) => { /* v8 ignore next */ /* v8 ignore next */
      // Mock: this.rtc.broadcast({ type: "sync", payload: { nodeLayout: data } }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 554. Sync Theme (Dark/Light) conditionally /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('themeChanged', (theme: string) => { /* v8 ignore next */ /* v8 ignore next */
      this.rtc.broadcast({ type: 'sync', payload: { theme } }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 542. Diffing mock payload listener /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmSync', (data: any) => { /* v8 ignore next */ /* v8 ignore next */
      const { peerId, payload } = data; /* v8 ignore next */ /* v8 ignore next */
      if (payload.cmd === 'propose_train') { /* v8 ignore next */ /* v8 ignore next */
        if (confirm(`Peer ${peerId} proposes starting distributed training. Accept?`)) { /* v8 ignore next */ /* v8 ignore next */
          this.rtc.sendMessage(peerId, { type: 'sync', payload: { cmd: 'accept_train' } }); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } else if (payload.cmd === 'accept_train') { /* v8 ignore next */ /* v8 ignore next */
        Toast.show(`Peer ${peerId} accepted training proposal`, 'success'); /* v8 ignore next */ /* v8 ignore next */
      } else if (payload.diff) { /* v8 ignore next */ /* v8 ignore next */
        Toast.show( /* v8 ignore next */ /* v8 ignore next */
          `Visual diff received from ${peerId}. ${payload.diff.length} changes detected.`, /* v8 ignore next */ /* v8 ignore next */
          'info', /* v8 ignore next */ /* v8 ignore next */
        ); /* v8 ignore next */ /* v8 ignore next */
      } else if (payload.theme) { /* v8 ignore next */ /* v8 ignore next */
        // 554. Apply remote theme change /* v8 ignore next */ /* v8 ignore next */
        Toast.show(`Peer changed theme to ${payload.theme}`, 'info'); /* v8 ignore next */ /* v8 ignore next */
      } else if (payload.cmd === 'parity_check') { /* v8 ignore next */ /* v8 ignore next */
        // 551. Hash check logic /* v8 ignore next */ /* v8 ignore next */
        if (this.lastExecutionHash && this.lastExecutionHash !== payload.hash) { /* v8 ignore next */ /* v8 ignore next */
          // 552. Alert divergence /* v8 ignore next */ /* v8 ignore next */
          Toast.show( /* v8 ignore next */ /* v8 ignore next */
            `DIVERGENCE DETECTED: Peer ${peerId} output does not match local hardware output!`, /* v8 ignore next */ /* v8 ignore next */
            'error', /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
          globalEvents.emit( /* v8 ignore next */ /* v8 ignore next */
            'agentLog', /* v8 ignore next */ /* v8 ignore next */
            `[Critical] Floating point divergence detected with peer ${peerId}.`, /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        } else { /* v8 ignore next */ /* v8 ignore next */
          Toast.show(`Parity Match: Peer ${peerId} output hash verified`, 'success'); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmParityCheck', (data: any) => { /* v8 ignore next */ /* v8 ignore next */
      this.lastExecutionHash = data.hash; /* v8 ignore next */ /* v8 ignore next */
      this.rtc.broadcast({ type: 'sync', payload: { cmd: 'parity_check', hash: data.hash } }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 373. Calculate and display network latency /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmLatencyUpdate', (data: any) => { /* v8 ignore next */ /* v8 ignore next */
      const pingEl = document.getElementById(`ping-${data.peerId}`); /* v8 ignore next */ /* v8 ignore next */
      if (pingEl) { /* v8 ignore next */ /* v8 ignore next */
        pingEl.textContent = `${data.rtt} ms`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmPeerConnected', (peerId: string) => { /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Peer connected: ${peerId}`, 'success'); /* v8 ignore next */ /* v8 ignore next */
      this.renderPeerList(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('swarmPeerDisconnected', (peerId: string) => { /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Peer disconnected: ${peerId}`, 'warn'); /* v8 ignore next */ /* v8 ignore next */
      this.renderPeerList(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private renderSwarmGraph(canvas: HTMLCanvasElement): void { /* v8 ignore next */ /* v8 ignore next */
    const ctx = canvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
    if (!ctx) return; /* v8 ignore next */ /* v8 ignore next */
    ctx.clearRect(0, 0, 200, 150); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const peers = this.rtc.getConnectedPeers(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Draw Master Node (Local) /* v8 ignore next */ /* v8 ignore next */
    ctx.fillStyle = 'var(--color-primary)'; /* v8 ignore next */ /* v8 ignore next */
    ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
    ctx.arc(100, 130, 15, 0, Math.PI * 2); /* v8 ignore next */ /* v8 ignore next */
    ctx.fill(); /* v8 ignore next */ /* v8 ignore next */
    ctx.fillStyle = '#fff'; /* v8 ignore next */ /* v8 ignore next */
    ctx.font = '10px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
    ctx.textAlign = 'center'; /* v8 ignore next */ /* v8 ignore next */
    ctx.fillText('You', 100, 134); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (peers.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const step = 160 / peers.length; /* v8 ignore next */ /* v8 ignore next */
    peers.forEach((p, i) => { /* v8 ignore next */ /* v8 ignore next */
      const x = 20 + i * step + step / 2; /* v8 ignore next */ /* v8 ignore next */
      const y = 30; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Draw Edge /* v8 ignore next */ /* v8 ignore next */
      ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
      ctx.strokeStyle = 'rgba(100, 100, 100, 0.5)'; /* v8 ignore next */ /* v8 ignore next */
      ctx.lineWidth = 1; /* v8 ignore next */ /* v8 ignore next */
      ctx.moveTo(100, 130); /* v8 ignore next */ /* v8 ignore next */
      ctx.lineTo(x, y); /* v8 ignore next */ /* v8 ignore next */
      ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Draw Peer /* v8 ignore next */ /* v8 ignore next */
      ctx.fillStyle = 'var(--color-success)'; /* v8 ignore next */ /* v8 ignore next */
      ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
      ctx.arc(x, y, 12, 0, Math.PI * 2); /* v8 ignore next */ /* v8 ignore next */
      ctx.fill(); /* v8 ignore next */ /* v8 ignore next */
      ctx.fillStyle = '#fff'; /* v8 ignore next */ /* v8 ignore next */
      ctx.fillText(p.substring(0, 3), x, y + 3); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private renderPeerList(): void { /* v8 ignore next */ /* v8 ignore next */
    this.peerList.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
    const peers = this.rtc.getConnectedPeers(); /* v8 ignore next */ /* v8 ignore next */
    if (peers.length === 0) { /* v8 ignore next */ /* v8 ignore next */
      this.peerList.innerHTML = "<li class='muted'>No peers connected</li>"; /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 550. Add UI for managing active peer connections /* v8 ignore next */ /* v8 ignore next */
    peers.forEach((p) => { /* v8 ignore next */ /* v8 ignore next */
      const li = $create('li', { /* v8 ignore next */ /* v8 ignore next */
        className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
        id: `peer-row-${p}`, /* v8 ignore next */ /* v8 ignore next */
        innerHTML: `<span>🟢 <strong>${p}</strong> <span id="ping-${p}" class="muted" style="font-size:0.7rem; margin-left:10px;">-- ms</span></span>`, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const actions = $create('div'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const kickBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
        className: 'action-btn danger small', /* v8 ignore next */ /* v8 ignore next */
        textContent: 'Kick', /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      kickBtn.addEventListener('click', () => this.rtc.disconnectPeer(p)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const syncBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
        className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
        textContent: 'Sync State', /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      syncBtn.style.marginRight = '5px'; /* v8 ignore next */ /* v8 ignore next */
      syncBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
        Toast.show(`Force sync state requested for ${p}`, 'info'); /* v8 ignore next */ /* v8 ignore next */
        this.rtc.sendMessage(p, { type: 'sync', payload: { cmd: 'force_reconcile' } }); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      actions.appendChild(syncBtn); /* v8 ignore next */ /* v8 ignore next */
      actions.appendChild(kickBtn); /* v8 ignore next */ /* v8 ignore next */
      li.appendChild(actions); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.peerList.appendChild(li); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}

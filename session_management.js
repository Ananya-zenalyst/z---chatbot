/**
 * Session Management for Chatbot Application
 *
 * This script provides:
 * 1. Automatic session ID generation and storage
 * 2. Tab/window focus detection for inactivity tracking
 * 3. Automatic cleanup when tab is inactive for 5 minutes
 * 4. Manual cleanup functionality
 */

class SessionManager {
    constructor() {
        this.sessionId = null;
        this.isActive = true;
        this.inactivityTimer = null;
        this.INACTIVITY_TIMEOUT = 5 * 60 * 1000; // 5 minutes in milliseconds
        this.API_BASE_URL = 'http://localhost:8000'; // Adjust based on your FastAPI server

        this.init();
    }

    init() {
        // Get or create session ID
        this.sessionId = this.getOrCreateSessionId();

        // Set up focus/blur event listeners
        this.setupActivityListeners();

        // Set up page visibility API
        this.setupVisibilityListener();

        // Set up beforeunload event for cleanup
        this.setupBeforeUnloadListener();

        console.log('Session Manager initialized with session ID:', this.sessionId);
    }

    getOrCreateSessionId() {
        let sessionId = sessionStorage.getItem('chatbot_session_id');
        if (!sessionId) {
            sessionId = this.generateSessionId();
            sessionStorage.setItem('chatbot_session_id', sessionId);
        }
        return sessionId;
    }

    generateSessionId() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    setupActivityListeners() {
        // Window focus/blur events
        window.addEventListener('focus', () => {
            this.onActivityResume();
        });

        window.addEventListener('blur', () => {
            this.onActivityPause();
        });

        // Mouse and keyboard activity
        const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];
        activityEvents.forEach(event => {
            document.addEventListener(event, () => {
                this.resetInactivityTimer();
            }, true);
        });
    }

    setupVisibilityListener() {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.onActivityPause();
            } else {
                this.onActivityResume();
            }
        });
    }

    setupBeforeUnloadListener() {
        window.addEventListener('beforeunload', () => {
            this.cleanupSession();
        });
    }

    onActivityResume() {
        this.isActive = true;
        this.clearInactivityTimer();
        console.log('Activity resumed for session:', this.sessionId);
    }

    onActivityPause() {
        this.isActive = false;
        this.startInactivityTimer();
        console.log('Activity paused for session:', this.sessionId);
    }

    startInactivityTimer() {
        this.clearInactivityTimer();

        this.inactivityTimer = setTimeout(() => {
            console.log('Session inactive for 5 minutes, cleaning up session:', this.sessionId);
            this.cleanupSession();
        }, this.INACTIVITY_TIMEOUT);
    }

    clearInactivityTimer() {
        if (this.inactivityTimer) {
            clearTimeout(this.inactivityTimer);
            this.inactivityTimer = null;
        }
    }

    resetInactivityTimer() {
        if (!this.isActive) {
            return; // Don't reset if tab is not visible
        }

        this.clearInactivityTimer();
        this.startInactivityTimer();
    }

    async cleanupSession() {
        if (!this.sessionId) {
            return;
        }

        try {
            const response = await fetch(`${this.API_BASE_URL}/delete/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });

            if (response.ok) {
                console.log('Session cleaned up successfully:', this.sessionId);
                sessionStorage.removeItem('chatbot_session_id');
                this.sessionId = null;
            } else {
                console.error('Failed to cleanup session:', response.statusText);
            }
        } catch (error) {
            console.error('Error cleaning up session:', error);
        }
    }

    async manualCleanup() {
        await this.cleanupSession();
        // Generate new session ID after cleanup
        this.sessionId = this.generateSessionId();
        sessionStorage.setItem('chatbot_session_id', this.sessionId);
        console.log('Manual cleanup completed, new session ID:', this.sessionId);

        // Notify the application about the new session
        this.notifySessionChange();
    }

    notifySessionChange() {
        // Dispatch custom event for Streamlit to listen to
        const event = new CustomEvent('sessionChanged', {
            detail: { sessionId: this.sessionId }
        });
        window.dispatchEvent(event);
    }

    getCurrentSessionId() {
        return this.sessionId;
    }

    async getSessionInfo() {
        if (!this.sessionId) {
            return null;
        }

        try {
            const response = await fetch(`${this.API_BASE_URL}/sessions/${this.sessionId}/info`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Error getting session info:', error);
        }
        return null;
    }

    async uploadFiles(files) {
        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch(`${this.API_BASE_URL}/upload/`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                // Update session ID with the one returned from upload
                this.sessionId = result.session_id;
                sessionStorage.setItem('chatbot_session_id', this.sessionId);
                console.log('Files uploaded, session ID updated:', this.sessionId);
                return result;
            } else {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Error uploading files:', error);
            throw error;
        }
    }

    async sendMessage(message) {
        if (!this.sessionId) {
            throw new Error('No active session. Please upload documents first.');
        }

        try {
            const response = await fetch(`${this.API_BASE_URL}/chat/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message,
                    session_id: this.sessionId
                })
            });

            if (response.ok) {
                return await response.json();
            } else {
                throw new Error(`Chat failed: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            throw error;
        }
    }
}

// Global session manager instance
window.sessionManager = new SessionManager();

// Utility functions for Streamlit integration
window.getSessionId = () => window.sessionManager.getCurrentSessionId();
window.cleanupSession = () => window.sessionManager.manualCleanup();
window.getSessionInfo = () => window.sessionManager.getSessionInfo();

console.log('Session Management script loaded successfully');
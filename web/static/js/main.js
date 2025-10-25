// VMEvalKit Dashboard - Interactive Features

'use strict';

// Configuration
const CONFIG = {
    VIDEO_LOAD_THRESHOLD: 100, // px
    SEARCH_DEBOUNCE_DELAY: 300, // ms
    NOTIFICATION_DURATION: 3000, // ms
    MAX_VIDEO_OBSERVERS: 50
};

// State management
let isInitialized = false;
let activeObservers = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    if (isInitialized) {
        console.warn('VMEvalKit dashboard already initialized');
        return;
    }
    
    try {
        initializeVideoPlayers();
        initializeFilters();
        initializeSearch();
        initializeLazyLoading();
        initializeAccessibility();
        initializeProgressBars();
        isInitialized = true;
        console.info('VMEvalKit dashboard initialized successfully');
    } catch (error) {
        console.error('Failed to initialize VMEvalKit dashboard:', error);
        showNotification('Dashboard initialization failed', 'error');
    }
});

// Video player enhancements
function initializeVideoPlayers() {
    const videos = document.querySelectorAll('video');
    
    if (!videos.length) {
        console.info('No videos found to initialize');
        return;
    }
    
    videos.forEach((video, index) => {
        try {
            // Skip if already initialized
            if (video.dataset.initialized === 'true') {
                return;
            }
            
            // Add unique identifier
            if (!video.id) {
                video.id = `video-${index}`;
            }
            
            // Add play/pause on click with error handling
            video.addEventListener('click', function(event) {
                event.preventDefault();
                toggleVideoPlayback(this);
            });
            
            // Add keyboard support
            video.addEventListener('keydown', function(event) {
                if (event.key === ' ' || event.key === 'Enter') {
                    event.preventDefault();
                    toggleVideoPlayback(this);
                }
            });
            
            // Add comprehensive loading indicators
            video.addEventListener('loadstart', function() {
                this.classList.add('loading');
                console.debug(`Video load started: ${this.id || this.src}`);
            });
            
            video.addEventListener('loadedmetadata', function() {
                console.debug(`Video metadata loaded: ${this.id || this.src}`);
            });
            
            video.addEventListener('loadeddata', function() {
                this.classList.remove('loading');
                console.debug(`Video data loaded: ${this.id || this.src}`);
            });
            
            video.addEventListener('canplay', function() {
                this.classList.add('can-play');
                console.debug(`Video can play: ${this.id || this.src}`);
            });
            
            video.addEventListener('canplaythrough', function() {
                this.classList.add('ready');
                console.debug(`Video ready for full playback: ${this.id || this.src}`);
            });
            
            // Handle errors with better error reporting
            video.addEventListener('error', function(event) {
                console.error(`Video failed to load (${this.id}):`, this.src, event);
                handleVideoError(this);
            });
            
            // Add ARIA labels for accessibility
            video.setAttribute('aria-label', 'Generated video result');
            video.setAttribute('role', 'button');
            video.setAttribute('tabindex', '0');
            
            // Mark as initialized
            video.dataset.initialized = 'true';
            
        } catch (error) {
            console.error(`Failed to initialize video ${index}:`, error);
        }
    });
    
    console.info(`Initialized ${videos.length} video players`);
}

// Helper function to toggle video playback
function toggleVideoPlayback(video) {
    try {
        if (video.paused) {
            video.play().catch(error => {
                console.error('Failed to play video:', error);
                showNotification('Failed to play video', 'error');
            });
        } else {
            video.pause();
        }
    } catch (error) {
        console.error('Error toggling video playback:', error);
        showNotification('Video control error', 'error');
    }
}

// Helper function to handle video errors
function handleVideoError(video) {
    const container = video.closest('.result-video') || 
                     video.closest('.comparison-section') || 
                     video.closest('.task-video-section');
    
    if (container) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'video-error';
        errorDiv.innerHTML = '❌ Video failed to load';
        errorDiv.setAttribute('role', 'alert');
        container.replaceChild(errorDiv, video);
    }
}

// Filter functionality for tables
function initializeFilters() {
    const filterInputs = document.querySelectorAll('[data-filter]');
    
    filterInputs.forEach(input => {
        input.addEventListener('input', function() {
            const filterValue = this.value.toLowerCase();
            const targetTable = document.querySelector(this.dataset.filter);
            
            if (targetTable) {
                const rows = targetTable.querySelectorAll('tbody tr');
                
                rows.forEach(row => {
                    const text = row.textContent.toLowerCase();
                    row.style.display = text.includes(filterValue) ? '' : 'none';
                });
            }
        });
    });
}

// Search functionality
function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', debounce(function() {
        const query = this.value.toLowerCase();
        const cards = document.querySelectorAll('.result-card, .domain-card, .comparison-card');
        
        cards.forEach(card => {
            const text = card.textContent.toLowerCase();
            card.style.display = text.includes(query) ? '' : 'none';
        });
    }, CONFIG.SEARCH_DEBOUNCE_DELAY));
}

// Enhanced video loading - now truly lazy, only loads when sections are expanded
function initializeLazyLoading() {
    try {
        // For videos that have been upgraded to metadata preload, enhance them further when visible
        if ('IntersectionObserver' in window) {
            // Function to observe videos that have metadata preloading
            const observeMetadataVideos = () => {
                const videos = document.querySelectorAll('video[preload="metadata"]');
                
                if (!videos.length) {
                    console.info('No metadata-preload videos found to enhance');
                    return;
                }
                
                // Create observer for full video preloading when in viewport
                const videoLoadObserver = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            try {
                                const video = entry.target;
                                
                                // Upgrade to full preloading for smooth playback
                                if (video.preload !== 'auto') {
                                    video.preload = 'auto';
                                    console.debug(`Enhanced preloading for video: ${video.id || video.src}`);
                                }
                                
                                // Load the video data
                                video.load();
                                
                                // Unobserve after loading starts
                                videoLoadObserver.unobserve(video);
                                
                            } catch (error) {
                                console.error('Error enhancing video loading:', error);
                            }
                        }
                    });
                }, {
                    rootMargin: '100px', // Start loading when video is 100px away from viewport
                    threshold: 0.1
                });
                
                videos.forEach(video => {
                    try {
                        videoLoadObserver.observe(video);
                    } catch (error) {
                        console.error('Failed to observe video for enhanced loading:', error);
                    }
                });
                
                // Store observer for cleanup
                activeObservers.push(videoLoadObserver);
                
                console.info(`Enhanced loading initialized for ${videos.length} metadata-preload videos`);
            };
            
            // Initial check for any existing metadata videos
            observeMetadataVideos();
            
            // Make the function globally available for manual triggering
            window.VMEvalKit = window.VMEvalKit || {};
            window.VMEvalKit.observeMetadataVideos = observeMetadataVideos;
            
        } else {
            // Fallback: just log that lazy loading is not available
            console.info('IntersectionObserver not available, videos will load when sections expand');
        }
        
    } catch (error) {
        console.error('Failed to initialize enhanced video loading:', error);
    }
}

// Initialize progress bars
function initializeProgressBars() {
    try {
        const progressFills = document.querySelectorAll('.progress-fill[data-width]');
        
        progressFills.forEach(fill => {
            const width = fill.getAttribute('data-width');
            if (width !== null && width !== '') {
                // Set width with animation
                setTimeout(() => {
                    fill.style.width = `${width}%`;
                }, 100);
            }
        });
        
        console.info(`Initialized ${progressFills.length} progress bars`);
    } catch (error) {
        console.error('Failed to initialize progress bars:', error);
    }
}

// Initialize accessibility features
function initializeAccessibility() {
    try {
        // Add skip to main content link
        addSkipToMainContent();
        
        // Enhance focus management
        enhanceFocusManagement();
        
        // Add ARIA live region for notifications
        addLiveRegion();
        
        console.info('Accessibility features initialized');
    } catch (error) {
        console.error('Failed to initialize accessibility features:', error);
    }
}

// Add skip to main content link
function addSkipToMainContent() {
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.className = 'skip-to-main';
    skipLink.textContent = 'Skip to main content';
    skipLink.style.cssText = `
        position: absolute;
        top: -40px;
        left: 6px;
        background: var(--primary, #000);
        color: var(--bg, #fff);
        padding: 8px;
        text-decoration: none;
        border-radius: 4px;
        z-index: 10001;
        transition: top 0.3s;
    `;
    
    skipLink.addEventListener('focus', () => {
        skipLink.style.top = '6px';
    });
    
    skipLink.addEventListener('blur', () => {
        skipLink.style.top = '-40px';
    });
    
    document.body.insertBefore(skipLink, document.body.firstChild);
}

// Enhance focus management
function enhanceFocusManagement() {
    // Track focus for keyboard navigation
    let isUsingKeyboard = false;
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            isUsingKeyboard = true;
        }
    });
    
    document.addEventListener('mousedown', () => {
        isUsingKeyboard = false;
    });
    
    // Add focus indicators only for keyboard navigation
    document.addEventListener('focusin', (e) => {
        if (isUsingKeyboard) {
            e.target.classList.add('keyboard-focus');
        }
    });
    
    document.addEventListener('focusout', (e) => {
        e.target.classList.remove('keyboard-focus');
    });
}

// Add ARIA live region for notifications
function addLiveRegion() {
    const liveRegion = document.createElement('div');
    liveRegion.id = 'aria-live-region';
    liveRegion.setAttribute('aria-live', 'polite');
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.style.cssText = `
        position: absolute;
        left: -10000px;
        width: 1px;
        height: 1px;
        overflow: hidden;
    `;
    document.body.appendChild(liveRegion);
}

// Utility: Debounce function with better type handling
function debounce(func, wait) {
    if (typeof func !== 'function') {
        throw new TypeError('Expected a function');
    }
    
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func.apply(this, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Copy to clipboard with fallback
function copyToClipboard(text) {
    if (!text || typeof text !== 'string') {
        console.error('Invalid text provided to copyToClipboard');
        showNotification('Nothing to copy', 'error');
        return;
    }
    
    if (navigator.clipboard && window.isSecureContext) {
        // Modern clipboard API
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Copied to clipboard!', 'success');
            announceToScreenReader('Copied to clipboard');
        }).catch(err => {
            console.error('Failed to copy with clipboard API:', err);
            fallbackCopyToClipboard(text);
        });
    } else {
        // Fallback for older browsers or non-secure contexts
        fallbackCopyToClipboard(text);
    }
}

// Fallback copy method
function fallbackCopyToClipboard(text) {
    try {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.cssText = `
            position: fixed;
            top: -1000px;
            left: -1000px;
            width: 2em;
            height: 2em;
            padding: 0;
            border: none;
            outline: none;
            box-shadow: none;
            background: transparent;
        `;
        
        document.body.appendChild(textArea);
        textArea.select();
        textArea.setSelectionRange(0, 99999); // For mobile devices
        
        const successful = document.execCommand('copy');
        document.body.removeChild(textArea);
        
        if (successful) {
            showNotification('Copied to clipboard!', 'success');
            announceToScreenReader('Copied to clipboard');
        } else {
            throw new Error('execCommand failed');
        }
    } catch (err) {
        console.error('Failed to copy with fallback method:', err);
        showNotification('Failed to copy to clipboard', 'error');
    }
}

// Announce to screen readers
function announceToScreenReader(message) {
    const liveRegion = document.getElementById('aria-live-region');
    if (liveRegion) {
        liveRegion.textContent = message;
        // Clear after a short delay to allow for repeated announcements
        setTimeout(() => {
            liveRegion.textContent = '';
        }, 1000);
    }
}

// Show notification with improved accessibility and error handling
function showNotification(message, type = 'info') {
    if (!message || typeof message !== 'string') {
        console.error('Invalid message provided to showNotification');
        return;
    }
    
    try {
        // Remove existing notifications of the same type
        const existingNotifications = document.querySelectorAll(`.notification-${type}`);
        existingNotifications.forEach(notification => notification.remove());
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.setAttribute('role', 'alert');
        notification.setAttribute('aria-live', 'assertive');
        
        const colors = {
            success: '#10b981',
            error: '#ef4444',
            warning: '#f59e0b',
            info: '#2563eb'
        };
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: ${colors[type] || colors.info};
            color: white;
            border-radius: 0.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 10000;
            font-weight: 500;
            max-width: 300px;
            word-wrap: break-word;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after configured duration
        const timeoutId = setTimeout(() => {
            if (document.body.contains(notification)) {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => {
                    if (document.body.contains(notification)) {
                        notification.remove();
                    }
                }, 300);
            }
        }, CONFIG.NOTIFICATION_DURATION);
        
        // Allow manual dismissal on click
        notification.addEventListener('click', () => {
            clearTimeout(timeoutId);
            notification.remove();
        });
        
        // Announce to screen readers
        announceToScreenReader(message);
        
    } catch (error) {
        console.error('Failed to show notification:', error);
    }
}

// Sort table columns
function sortTable(table, column, ascending = true) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort((a, b) => {
        const aText = a.cells[column].textContent.trim();
        const bText = b.cells[column].textContent.trim();
        
        // Try to parse as numbers
        const aNum = parseFloat(aText);
        const bNum = parseFloat(bText);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return ascending ? aNum - bNum : bNum - aNum;
        }
        
        // String comparison
        return ascending ? 
            aText.localeCompare(bText) : 
            bText.localeCompare(aText);
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K for search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('search-input');
        if (searchInput) searchInput.focus();
    }
    
    // Escape to clear search
    if (e.key === 'Escape') {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.value = '';
            searchInput.dispatchEvent(new Event('input'));
        }
    }
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
    
    .video-error {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 200px;
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        font-weight: 600;
    }
    
    video.loading {
        opacity: 0.5;
    }
`;
document.head.appendChild(style);

// Cleanup function for page unload
function cleanup() {
    try {
        // Disconnect all intersection observers
        activeObservers.forEach(observer => {
            if (observer && typeof observer.disconnect === 'function') {
                observer.disconnect();
            }
        });
        activeObservers.length = 0;
        
        // Pause all videos
        const videos = document.querySelectorAll('video');
        videos.forEach(video => {
            try {
                video.pause();
            } catch (e) {
                // Ignore errors when pausing videos
            }
        });
        
        console.info('VMEvalKit dashboard cleanup completed');
    } catch (error) {
        console.error('Error during cleanup:', error);
    }
}

// Register cleanup on page unload
window.addEventListener('beforeunload', cleanup);
window.addEventListener('pagehide', cleanup);

// Export functions for use in templates
window.VMEvalKit = {
    copyToClipboard,
    showNotification,
    sortTable,
    cleanup
};


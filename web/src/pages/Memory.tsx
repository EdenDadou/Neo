import { useState } from 'react';
import { api } from '../lib/api';

interface MemoryItem {
  id: string;
  type: string;
  content: string;
  importance: number;
  createdAt: string;
}

export function Memory() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<MemoryItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  // Form for adding memory
  const [newType, setNewType] = useState<'fact' | 'preference' | 'skill'>('fact');
  const [newContent, setNewContent] = useState('');
  const [newTags, setNewTags] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [addSuccess, setAddSuccess] = useState(false);

  // Stats
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const results = await api.searchMemory(searchQuery) as MemoryItem[];
      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleAddMemory = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newContent.trim()) return;

    setIsAdding(true);
    setAddSuccess(false);

    try {
      const tags = newTags.split(',').map(t => t.trim()).filter(Boolean);
      await api.storeMemory(newType, newContent, tags.length > 0 ? tags : undefined);
      setAddSuccess(true);
      setNewContent('');
      setNewTags('');
      setTimeout(() => setAddSuccess(false), 3000);
    } catch (error) {
      console.error('Add memory error:', error);
    } finally {
      setIsAdding(false);
    }
  };

  const loadStats = async () => {
    try {
      const data = await api.getStats();
      setStats(data as Record<string, unknown>);
    } catch (error) {
      console.error('Stats error:', error);
    }
  };

  return (
    <div className="memory-container">
      <div className="memory-header">
        <h1>🧠 Memory</h1>
        <p>Explorez et gérez la mémoire de votre AI</p>
      </div>

      <div className="memory-grid">
        {/* Search Section */}
        <section className="memory-section">
          <h2>🔍 Rechercher</h2>
          <form onSubmit={handleSearch} className="search-form">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Rechercher dans la mémoire..."
            />
            <button type="submit" disabled={isSearching}>
              {isSearching ? 'Recherche...' : 'Rechercher'}
            </button>
          </form>

          {searchResults.length > 0 && (
            <div className="search-results">
              <h3>Résultats ({searchResults.length})</h3>
              {searchResults.map(item => (
                <div key={item.id} className="memory-item">
                  <div className="memory-type">{item.type}</div>
                  <div className="memory-content">{item.content}</div>
                  <div className="memory-meta">
                    Importance: {(item.importance * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Add Memory Section */}
        <section className="memory-section">
          <h2>➕ Ajouter une mémoire</h2>
          <form onSubmit={handleAddMemory} className="add-memory-form">
            <div className="form-group">
              <label>Type</label>
              <select
                value={newType}
                onChange={(e) => setNewType(e.target.value as 'fact' | 'preference' | 'skill')}
              >
                <option value="fact">Fait</option>
                <option value="preference">Préférence</option>
                <option value="skill">Skill</option>
              </select>
            </div>

            <div className="form-group">
              <label>Contenu</label>
              <textarea
                value={newContent}
                onChange={(e) => setNewContent(e.target.value)}
                placeholder="Entrez l'information à mémoriser..."
                rows={3}
              />
            </div>

            <div className="form-group">
              <label>Tags (séparés par des virgules)</label>
              <input
                type="text"
                value={newTags}
                onChange={(e) => setNewTags(e.target.value)}
                placeholder="tag1, tag2, tag3"
              />
            </div>

            <button type="submit" disabled={isAdding || !newContent.trim()}>
              {isAdding ? 'Ajout...' : 'Ajouter'}
            </button>

            {addSuccess && (
              <div className="success-message">
                ✅ Mémoire ajoutée avec succès!
              </div>
            )}
          </form>
        </section>

        {/* Stats Section */}
        <section className="memory-section">
          <h2>📊 Statistiques</h2>
          <button onClick={loadStats} className="load-stats-btn">
            Charger les stats
          </button>

          {stats && (
            <div className="stats-display">
              <pre>{JSON.stringify(stats, null, 2)}</pre>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

// Shared request type (mirrors backend PlayInput)
export type SeasonType = 'REG' | 'POST';

export interface PlayInput {
  qtr: number; // 1..5
  quarter_seconds_remaining?: number; // 0..900
  game_seconds_remaining?: number; // optional; derived if not provided
  yardline_100: number; // 1..99
  ydstogo: number; // >=1
  score_differential: number; // posteam - defteam
  posteam_timeouts_remaining: number; // 0..3
  defteam_timeouts_remaining: number; // 0..3

  season_type?: SeasonType; // default REG if omitted
  roof?: string; // raw string (mapped server-side)
  surface?: string; // raw string (mapped server-side)
  temp_f?: number; // optional
  wind_mph?: number; // optional
}

// Responses
export interface WpResponse {
  go_wp: number;
  fg_wp: number;
  punt_wp: number;
  best_action: 'GO' | 'FIELD_GOAL' | 'PUNT';
}

export interface CompResponse {
  fg_make_prob: number;
  first_down_prob: number;
}

export interface CoachResponse {
  policy: 'GO' | 'FIELD_GOAL' | 'PUNT';
  probs: Record<'GO' | 'FIELD_GOAL' | 'PUNT', number>;
}

export interface AllResponse {
  wp: WpResponse;
  comp: CompResponse;
  coach: CoachResponse;
}

